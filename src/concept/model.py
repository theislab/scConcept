import os
from typing import Optional
import wandb
import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from itertools import combinations
import random
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
from torchmetrics.functional.regression import r2_score
import numpy as np
from torch.distributed.nn.functional import all_gather
from collections import defaultdict
from functools import partial
from concept.modules.bert_padding import unpad_input, pad_input
from concept.modules.flash_attention_layer import FlashTransformerEncoderLayer
from concept.modules.transformer import TransformerEncoder


# set random seed
random.seed(42)

class BaseTransformerModel(L.LightningModule):
    MASK_VALUE = -1
    CLS_VALUE = -2

    def __init__(
        self,
        config,
        pad_token_id: int,
        cls_token_id: int,
        debug: bool = False,
    ):
        super().__init__()
        self.debug = debug
        self.flash_attention = config['flash_attention']
        self.dim_model = config['dim_model']
        self.dim_hid = config['dim_hid']
        self.num_head = config['num_head']
        self.nlayers = config['nlayers']
        self.dropout = config['dropout']
        self.decoder_head = config['decoder_head']
        self.mask_padding = config['mask_padding']
        self.input_encoding = config['input_encoding']
        self.PAD_TOKEN_ID = pad_token_id
        self.CLS_TOKEN_ID = cls_token_id
        self.masking_rate = config['training']['masking_rate']
        self.lr = config['training']['lr']
        self.weight_decay = config['training']['weight_decay']
        self.optimizer_class = config['training']['optimizer_class']
        self.scheduler = config['training']['scheduler']
        self.warmup = config['training']['warmup']
        self.max_steps = config['training']['max_steps']
        self.min_lr = config['training']['min_lr']
        self.values_only_sanity_check = config['values_only_sanity_check']
        self.data_loading_speed_sanity_check = config['data_loading_speed_sanity_check']

        encoder_layers = FlashTransformerEncoderLayer(
            self.dim_model, self.num_head, self.dim_hid, self.dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)
        # self.transformer_encoder = torch.compile(self.transformer_encoder) #todo: check compilation

        if self.decoder_head:
            self.expression_decoder = GeneExpressionDecoder(self.dim_model)

    def _encode(
        self,
        tokens: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
    ) -> Tensor:

        batch_size = tokens.size(0)
        tokens, indices, cu_seqlens, max_seqlen, seqlens = unpad_input(tokens.unsqueeze(-1), ~src_key_padding_mask)
        tokens = tokens.squeeze(-1)
        seqlens = list(seqlens)
        
        # seqlens = [len(t) for t in tokens]
        # cu_seqlens = torch.cumsum(torch.tensor([0] + seqlens, device=self.device, dtype=torch.int32), dim=0, dtype=torch.int32)
        # max_seqlen = max([len(t) for t in tokens])
        # tokens = torch.cat(tokens, dim=0)
        
        gene_embs = self._encode_gene_tokens(tokens) # (total_len, dim_model)
        
        if self.input_encoding == 'rank_encoding':
            total_embs = self.positional_encoder(gene_embs, seqlens=seqlens)
        elif self.input_encoding == 'value_encoding':
            values, _, _, _, _ = unpad_input(values.unsqueeze(-1), ~src_key_padding_mask)
            values = values.squeeze(-1)
            value_embs = self._encode_values(values)
            total_embs = gene_embs + value_embs
            # total_embs = torch.cat([gene_embs, value_embs], dim=-1)

        embs_jagged = self.transformer_encoder(total_embs, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        cell_embs_jagged = embs_jagged[cu_seqlens[:-1]]
        embs_padded = pad_input(embs_jagged, indices, batch_size, max_seqlen)
        cell_embs = embs_padded[:, 0, :]
        assert torch.equal(cell_embs_jagged, cell_embs), "cell_embs_jagged and cell_embs are not the same"
    
        
        # output = output.split(seqlens, dim=0)
        # assert [len(o) for o in output] == seqlens
        ############################## 3D ###############
        # gene_embs = self._encode_gene_tokens(tokens) # (batch, seq_len, dim_model)
        # total_embs = self.positional_encoder(gene_embs)
        # output = self.transformer_encoder(
        #     total_embs, key_padding_mask=src_key_padding_mask
        #     # total_embs, 
        # )
        #############################################
        return embs_padded, cell_embs

    def _encode_gene_tokens(self, tokens: Tensor) -> Tensor:
        raise NotImplementedError(
            "Subclasses must implement _encode_gene_tokens method.")

    def _encode_values(self, values: Tensor) -> Tensor:
        raise NotImplementedError(
            "Subclasses must implement _encode_values method.")

    def forward(self, input_tokens, input_values, src_key_padding_mask: Tensor = None):
        
        embs_padded, cell_embs = self._encode(input_tokens, input_values, src_key_padding_mask=src_key_padding_mask)
        pred = self.expression_decoder(embs_padded) if self.decoder_head else None
        
        return pred, embs_padded, cell_embs

    def _step(self, batch):
        raise NotImplementedError(
            "Subclasses must implement _step method.")

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val/loss", loss, sync_dist=True)

    def configure_optimizers(self):
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay) # default Adam
        if self.optimizer_class == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            
        if self.scheduler:
            if self.scheduler == 'warmup':
                lr_scheduler = WarmupScheduler(optimizer, warmup=self.warmup)
            elif self.scheduler == 'warmup_cosine':
                lr_scheduler = CosineWarmupScheduler(optimizer,
                                                warmup=self.warmup,
                                                max_steps=self.max_steps,
                                                min_lr=self.min_lr
                                                )
            
            return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
        else:
            return optimizer

    def mask_values(self, values: Tensor, rate: float, ignore_idxs=None) -> Tensor:
        mask = torch.rand(values.shape) < rate
        if ignore_idxs is not None:
            mask[:, ignore_idxs] = False
        masked_values = values.clone()
        masked_values[mask] = self.MASK_VALUE
        return masked_values, mask

    def _mlm_loss(self, pred, target, mask=None):
        if mask is not None:
            if torch.any(mask):
                pred = pred[mask]
                target = target[mask]
            else:
                return torch.tensor(0.0, device=self.device)
        return F.mse_loss(pred.float(), target.float())


class StandardModel(BaseTransformerModel):
    def __init__(
        self,
        vocab_size: int,
        dim_model: int,
        num_head: int,
        dim_hid: int,
        nlayers: int,
        pad_token_id: int,
        cls_token_id: int,
        masking_rate: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        dropout: float = 0.2,
    ):
        super().__init__(dim_model,
                         num_head,
                         dim_hid,
                         nlayers,
                         pad_token_id,
                         cls_token_id,
                         masking_rate,
                         lr,
                         weight_decay,
                         dropout)
        self.vocab_size = vocab_size

        self.gene_token_encoder = GeneEncoder(
            self.vocab_size, self.dim_model, padding_idx=None
        )

        self.value_encoder = ContinuousValueEncoder(
            self.dim_model, dropout=0.0)

    def _step(self, batch):
        input_tokens, input_values = batch['tokens'], batch['values']
        input_values_masked, masked_positions = self.mask_values(input_values, self.masking_rate)
        pred = self(input_tokens, input_values_masked)
        
        loss = self._mlm_loss(pred, input_values, masked_positions)
        return loss

    def _encode_gene_tokens(self, tokens: Tensor) -> Tensor:
        return self.gene_token_encoder(tokens)

    def _encode_values(self, values: Tensor) -> Tensor:
        return self.value_encoder(values)


class GeneEncoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        emb_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            n_genes, emb_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x


class ContinuousValueEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x.float().unsqueeze(-1)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize the Positional Encoding module.
        
        Args:
            d_model (int): Dimensionality of the model (i.e., the embedding size).
            max_len (int): Maximum length of the input sequences.
        """
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix to store the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Shape: (d_model/2,)

        # Sinusoidal function for even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Cosine function for odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)  # Register as a buffer (non-trainable parameter)

    def forward(self, x: torch.Tensor, seqlens=None) -> torch.Tensor:
        """
        Forward pass for the positional encoding.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            torch.Tensor: Input tensor with positional encodings added.
        """
        if x.dim() == 3:
            seq_len = x.size(1)
            # Add positional encoding to input tensor
            x = x + self.pe[:, :seq_len, :].to(x.device)
        elif x.dim() == 2:
            assert seqlens is not None, "Sequence lengths must be provided for 2D input tensor."
            arange = partial(torch.arange, device=x.device)
            indices = torch.cat([arange(l) for l in seqlens])
            x = x + self.pe[0, indices, :].to(x.device)
            
        return x


class GeneExpressionDecoder(nn.Module):
    def __init__(self, dim_model: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.LeakyReLU(),
            nn.Linear(dim_model, dim_model),
            nn.LeakyReLU(),
            nn.Linear(dim_model, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x).squeeze(-1)


class BiEncoderContrastiveModel(BaseTransformerModel):
    def __init__(
        self,
        config,
        pad_token_id: int,
        cls_token_id: int,
        vocab_size: int,
        precomp_embs_key: str = None,
        world_size: int = 1,
        val_loader_names = [],
        debug: bool = False,
    ):
        
        if config['mlm_loss_weight'] > 0:
            assert config['decoder_head'] == True, "Decoder head must be enabled for MLM loss"
        
        super().__init__(config,
                         pad_token_id,
                         cls_token_id,
                         debug=debug
                         )

        self.mlm_loss_weight = config['mlm_loss_weight']
        self.cont_loss_weight = config['cont_loss_weight']
        self.contrastive_loss = config['contrastive_loss']
        self.loss_switch_step = config['loss_switch_step']
        self.per_view_normalization = config['per_view_normalization']
        self.logit_scale_init_value = config['logit_scale_init_value']
        self.random_split = config['random_split']
        self.projection_dim = config['projection_dim']
        self.pe_max_len = config['pe_max_len']
        self.precomp_embs_key = precomp_embs_key
        self.vocab_size = vocab_size
        self.world_size = world_size
        self.val_loader_names = val_loader_names
        assert self.contrastive_loss in ['binary', 'multiclass']

        self.gene_token_encoder = GeneEncoder(self.vocab_size, self.dim_model, padding_idx=None)
        self.value_encoder = ContinuousValueEncoder(self.dim_model, dropout=0.0)
        self.positional_encoder = PositionalEncoding(self.dim_model, max_len=self.pe_max_len)

        self.binarcy_accuracy = BinaryAccuracy()
        self.logit_scale = nn.Parameter(torch.tensor(float(self.logit_scale_init_value)), requires_grad=True)
        if self.projection_dim:
            self.projection = nn.Linear(self.dim_model, self.projection_dim, bias=False)
            
        self.context_sizes = {'train': [], 'val': defaultdict(list)}
        self.logit_masks = {}
        
    def _encode_gene_tokens(self, tokens: Tensor) -> Tensor:
        return self.gene_token_encoder(tokens)

    def _encode_values(self, values: Tensor) -> Tensor:
        return self.value_encoder(values)

    def _step(self, batch, batch_idx, stage='train', random_split=False):
        assert stage in ['train', 'val'], f"Invalid stage: {stage}"
        
        # batch_1, batch_2 = self.split_inputs(batch, random_split)
        batch_1 = {'tokens': batch['tokens_1'], 'values': batch['values_1'], 'panel': batch['panel_1']}
        batch_2 = {'tokens': batch['tokens_2'], 'values': batch['values_2'], 'panel': batch['panel_2']}

        
        panel_size_1 = batch_1['panel'].shape[1]
        panel_size_2 = batch_2['panel'].shape[1]
        context_size_1 = len(batch_1['tokens'][0])
        context_size_2 = len(batch_2['tokens'][0])
        nonzero_cnt_1 = (batch_1['tokens'][0] != self.PAD_TOKEN_ID).sum().item()
        nonzero_cnt_2 = (batch_2['tokens'][0] != self.PAD_TOKEN_ID).sum().item()
        # if self.world_size > 1:
        #     nonzero_cnt_1 = torch.cat(all_gather(nonzero_cnt_1), dim=0)
        #     nonzero_cnt_2 = torch.cat(all_gather(nonzero_cnt_2), dim=0)
        try:
            min_1, min_2 = batch_1['values'][0].min().item(), batch_2['values'][0].min().item()
            max_1, max_2 = batch_1['values'][0].max().item(), batch_2['values'][0].max().item()
        except:
            min_1, min_2 = 0, 0
            max_1, max_2 = 0, 0
        
        panel_intersect = len(np.intersect1d(batch_1['panel'][0].cpu().numpy(), batch_2['panel'][0].cpu().numpy()))
        token_intersect = len(np.intersect1d(batch_1['tokens'][0].cpu().numpy(), batch_2['tokens'][0].cpu().numpy()))
        
        # if stage == 'train':    
        #     print(context_size_1, context_size_2)
        #     assert context_size_1[0] == context_size_1[1], f"Context size mismatch: {context_size_1}"
        #     assert context_size_2[0] == context_size_2[1], f"Context size mismatch: {context_size_2}"
        
        context_sizes = [
            batch['panel_name'],
            int(panel_size_1),
            int(panel_size_2),
            int(context_size_1),
            int(context_size_2),
            nonzero_cnt_1,
            nonzero_cnt_2,
            min_1, min_2,
            max_1, max_2,
            panel_intersect,
            token_intersect,
            ]
        
        if self.debug and batch_idx < 5 and stage == 'train':
            print(context_sizes)
            print('batch_1 values:', batch_1['values'][0])
            print('batch_2 values:', batch_2['values'][0])
        

        if self.values_only_sanity_check:
            batch_1['values'] = batch_1['values'][:, torch.randperm(batch_1['tokens'].size(1))]
            batch_2['values'] = batch_2['values'][:, torch.randperm(batch_2['tokens'].size(1))]

        
        if self.per_view_normalization:
            batch_1['values'] = self.normalize(batch_1['values'])
            batch_2['values'] = self.normalize(batch_2['values'])

        batch_1 = self.add_cls_token(batch_1)
        batch_2 = self.add_cls_token(batch_2)
        

        if self.masking_rate > 0:
            batch_1['values_masked'], batch_1['masked_positions'] = self.mask_values(
                batch_1['values'], self.masking_rate, ignore_idxs=[0])
            batch_2['values_masked'], batch_2['masked_positions'] = self.mask_values(
                batch_2['values'], self.masking_rate, ignore_idxs=[0])
        else:
            batch_1['values_masked'] = batch_1['values']
            batch_2['values_masked'] = batch_2['values']
            
        
        padding_mask_1 = (batch_1['tokens'] == self.PAD_TOKEN_ID) if self.mask_padding else torch.zeros_like(batch_1['tokens'], dtype=torch.bool, device=self.device)
        padding_mask_2 = (batch_2['tokens'] == self.PAD_TOKEN_ID) if self.mask_padding else torch.zeros_like(batch_2['tokens'], dtype=torch.bool, device=self.device)

        pred_1, embs_1, cell_embs_1 = self(batch_1['tokens'], batch_1['values_masked'], src_key_padding_mask=padding_mask_1)
        pred_2, embs_2, cell_embs_2 = self(batch_2['tokens'], batch_2['values_masked'], src_key_padding_mask=padding_mask_2)

        loss_mlm = torch.tensor(0.0, device=self.device)
        if self.mlm_loss_weight > 0:
            loss_mlm = self._mlm_loss(pred_1, batch_1['values'], batch_1['masked_positions']) + self._mlm_loss(
                pred_2, batch_2['values'], batch_2['masked_positions'])
            
        
        # Gather embeddings from GPUs and concatenate them
        if self.world_size > 1:
            cell_embs_1 = torch.cat(all_gather(cell_embs_1), dim=0)
            cell_embs_2 = torch.cat(all_gather(cell_embs_2), dim=0)

        if self.projection_dim:
            cell_embs_1 = self.projection(cell_embs_1)
            cell_embs_2 = self.projection(cell_embs_2)

        
        if self.precomp_embs_key and self.precomp_embs_key in batch:
            cell_embs_2 = torch.cat(all_gather(batch[self.precomp_embs_key]), dim=0)
            
            logits = (1.0 / (torch.cdist(cell_embs_1, cell_embs_2, p=2) + 1e-4)) * self.logit_scale.exp()
            
            cell_embs_concat_1 = torch.concat([cell_embs_1, cell_embs_2], dim=0)
            cell_embs_concat_2 = torch.concat([cell_embs_2, cell_embs_1], dim=0)
            logits_both_batch = (1.0 / (torch.cdist(cell_embs_concat_1, cell_embs_concat_2, p=2) + 1e-4)) * self.logit_scale.exp()

        else:
            cell_embs_1 = F.normalize(cell_embs_1, p=2, dim=1)
            cell_embs_2 = F.normalize(cell_embs_2, p=2, dim=1)
        
            logits = torch.mm(cell_embs_1, cell_embs_2.t()) * self.logit_scale.exp()
            
            cell_embs_concat_1 = torch.concat([cell_embs_1, cell_embs_2], dim=0)
            cell_embs_concat_2 = torch.concat([cell_embs_2, cell_embs_1], dim=0)
            logits_both_batch = torch.mm(cell_embs_concat_1, cell_embs_concat_2.t()) * self.logit_scale.exp()
            
        logit_size = len(logits_both_batch)
        logits_both_batch *= torch.roll(1 - torch.eye(logit_size, device=self.device), logit_size // 2, 1)

        
        # if stage == 'train':
        #     length_sim = torch.mm(nonzero_cnt_1.unsqueeze(1).float(), nonzero_cnt_2.unsqueeze(0).float())
        #     length_logit_corr = torch.tensor([torch.corrcoef(torch.stack([logits[i], length_sim[i]]))[0, 1] for i in range(logits.size(0))])
        #     length_logit_corr = torch.mean(length_logit_corr[~torch.isnan(length_logit_corr)]).item()
        #     self.log(f"{stage}/length_logit_corr", length_logit_corr, sync_dist=True)


        if self.debug and self.world_size == 1 and self.global_rank == 0 and stage == 'train' and batch_idx % 1000 == 0:
            print('Argmax: ', logits_both_batch.argmax(dim=1))
            print(f'batch_1["tokens"][0]: {batch_1["tokens"][0]}')
            print(f'batch_1["values"][0]: {batch_1["values"][0]}')
            print(f'batch_2["tokens"][0]: {batch_2["tokens"][0]}')
            print(f'batch_2["values"][0]: {batch_2["values"][0]}')

            idx = int(logits_both_batch.argmax(dim=1)[0])
            if idx < logit_size //2:
                print(f'match in batch_2, {idx} tokens: ', batch_2['tokens'][idx])
                print(f'match in batch_2, {idx} values: ', batch_2['values'][idx])
            else:
                idx = idx - logit_size //2
                print(f'match in batch_1, {idx} tokens: ', batch_1['tokens'][idx])
                print(f'match in batch_1, {idx} values: ', batch_1['values'][idx])

        if self.contrastive_loss == 'binary':
            loss_cont, acc_cont = self._contrastive_binary_loss(logits)
            top5_acc_cont = 0
            loss_cont_both_batch, acc_cont_both_batch, top5_acc_cont_both_batch = self._contrastive_binary_loss(logits_both_batch)
        elif self.contrastive_loss == 'multiclass':
            loss_cont, acc_cont, top5_acc_cont = self._clip_loss(logits)
            loss_cont_both_batch, acc_cont_both_batch, top5_acc_cont_both_batch = self._contrastive_multiclass_loss(logits_both_batch)
        
        if self.global_step < self.loss_switch_step:
            loss = self.mlm_loss_weight * loss_mlm + self.cont_loss_weight * loss_cont
        else:
            loss = self.mlm_loss_weight * loss_mlm + self.cont_loss_weight * loss_cont_both_batch
        
        if stage != 'train' and 'cell_type' in batch:
            labels_1, labels_2 = batch['cell_type'], batch['cell_type']
            if self.world_size > 1:
                labels_1 = torch.cat(all_gather(labels_1), dim=0)
                labels_2 = torch.cat(all_gather(labels_2), dim=0)
            label_acc = 0.5 * (self._knn_accuracy(logits, labels_1, labels_2) + self._knn_accuracy(logits.t(), labels_2, labels_1))
        else:
            label_acc = 0.0
            
        # if stage == 'train' and batch_idx % 100 == 0:
        #     length_r2 = 0.5 * (self._knn_r2(logits, nonzero_cnt_1, nonzero_cnt_2) + self._knn_r2(logits.t(), nonzero_cnt_2, nonzero_cnt_1))
        #     self.log(f"length_r2/{stage}", length_r2, sync_dist=True)
        
        if stage == 'train' and batch_idx % 100 == 0:
            global_rank = torch.tensor([self.global_rank]*len(batch_1['tokens']), device=self.device)
            if self.world_size > 1:
                global_rank = torch.cat(all_gather(global_rank), dim=0)
            global_rank_acc = 0.5 * (self._knn_accuracy(logits, global_rank, global_rank) + self._knn_accuracy(logits.t(), global_rank, global_rank))
            self.log(f"{stage}/global_rank_acc", global_rank_acc, sync_dist=True)
        
        if stage != 'train':
            same_batch_score = self._same_batch_score(logits_both_batch, k=1)
            same_batch_score_top_5 = self._same_batch_score(logits_both_batch, k=5)
        else:
            same_batch_score = 0.0
            same_batch_score_top_5 = 0.0
                
        
        return loss, loss_mlm, loss_cont, acc_cont, top5_acc_cont, loss_cont_both_batch, acc_cont_both_batch, top5_acc_cont_both_batch, label_acc, same_batch_score, same_batch_score_top_5, context_sizes

    def training_step(self, batch, batch_idx):
        if self.data_loading_speed_sanity_check:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            self.log("train/loss", loss, sync_dist=True)
            return loss
        
        loss, loss_mlm, loss_cont, acc_cont, top5_acc_cont, loss_cont_both_batch, acc_cont_both_batch, top5_acc_cont_both_batch, label_acc, same_batch_score, same_batch_score_top_5, context_sizes = self._step(batch, batch_idx, stage='train', random_split=self.random_split)
        self.context_sizes['train'].append(context_sizes)
        self.log(f"train/label_acc", label_acc, sync_dist=True)
        self.log(f"train/acc_cont_{self.contrastive_loss}", acc_cont, sync_dist=True)
        self.log(f"train/acc_top5_cont_{self.contrastive_loss}", top5_acc_cont, sync_dist=True)
        self.log(f"train/acc_cont_both_batch_{self.contrastive_loss}", acc_cont_both_batch, sync_dist=True)
        self.log(f"train/acc_top5_cont_both_batch_{self.contrastive_loss}", top5_acc_cont_both_batch, sync_dist=True)
        self.log("train/loss_mlm", loss_mlm, sync_dist=True)
        self.log("train/loss_cont", loss_cont, sync_dist=True)
        self.log("train/loss", loss, sync_dist=True)
        
        storage = torch.cat(all_gather(batch['dataset_id']), dim=0) if self.world_size > 1 else batch['dataset_id']
        if (storage[0] == storage).all():
            self.log('train/same_storage', 1, sync_dist=True, add_dataloader_idx=False) # todo: remove
        else:
            self.log('train/same_storage', 0, sync_dist=True, add_dataloader_idx=False)
            if self.debug and batch_idx < 3:
                print('storage: ', storage)


        if 'donor_id' in batch:
            donor = torch.cat(all_gather(batch['donor_id']), dim=0) if self.world_size > 1 else batch['donor_id']
            if (donor[0] == donor).all():
                self.log('train/same_donor', 1, sync_dist=True, add_dataloader_idx=False) # todo: remove
            else:
                self.log('train/same_donor', 0, sync_dist=True, add_dataloader_idx=False)
        
        if self.debug and 'panel_1' in batch and 'panel_2' in batch and batch_idx % 100 == 0:
            self._validate_panels(batch['panel_1'], batch['panel_2'])

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.data_loading_speed_sanity_check:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            self.log("val/loss", loss, sync_dist=True)
            return loss
        val_name = self.val_loader_names[dataloader_idx]
        loss, loss_mlm, loss_cont, acc_cont, top5_acc_cont, loss_cont_both_batch, acc_cont_both_batch, top5_acc_cont_both_batch, label_acc, same_batch_score, same_batch_score_top_5, context_sizes = self._step(batch, batch_idx, stage='val', random_split=False)
        self.context_sizes['val'][val_name].append(context_sizes)
        prefix = prefix=f'val/{val_name}' if val_name != 'same' else 'val'
        self.log(f"{prefix}/label_acc", label_acc, sync_dist=True, add_dataloader_idx=False)
        self.log(f"{prefix}/same_batch_score", same_batch_score, sync_dist=True, add_dataloader_idx=False)
        self.log(f"{prefix}/same_batch_score_top_5", same_batch_score_top_5, sync_dist=True, add_dataloader_idx=False)
        self.log(f"{prefix}/acc_cont_{self.contrastive_loss}", acc_cont, sync_dist=True, add_dataloader_idx=False)
        self.log(f"{prefix}/acc_top5_cont_{self.contrastive_loss}", top5_acc_cont, sync_dist=True, add_dataloader_idx=False)
        self.log(f"{prefix}/acc_cont_both_batch_{self.contrastive_loss}", acc_cont_both_batch, sync_dist=True, add_dataloader_idx=False)
        self.log(f"{prefix}/acc_top5_cont_both_batch_{self.contrastive_loss}", top5_acc_cont_both_batch, sync_dist=True, add_dataloader_idx=False)
        self.log(f"{prefix}/loss_mlm", loss_mlm, sync_dist=True, add_dataloader_idx=False)
        self.log(f"{prefix}/loss_cont", loss_cont, sync_dist=True, add_dataloader_idx=False)
        self.log(f"{prefix}/loss", loss, sync_dist=True, add_dataloader_idx=False)
        
        storage = torch.cat(all_gather(batch['dataset_id']), dim=0) if self.world_size > 1 else batch['dataset_id']
        if (storage[0] == storage).all():
            self.log(f'{prefix}/same_storage', 1, sync_dist=True, add_dataloader_idx=False) # todo: remove
        else:
            self.log(f'{prefix}/same_storage', 0, sync_dist=True, add_dataloader_idx=False)

            
        if 'donor_id' in batch:
            donor = torch.cat(all_gather(batch['donor_id']), dim=0) if self.world_size > 1 else batch['donor_id']
            if (donor[0] == donor).all():
                self.log(f'{prefix}/same_donor', 1, sync_dist=True, add_dataloader_idx=False) # todo: remove
            else:
                self.log(f'{prefix}/same_donor', 0, sync_dist=True, add_dataloader_idx=False)
        
        if self.debug and 'panel_1' in batch and 'panel_2' in batch and batch_idx % 100 == 0:
            self._validate_panels(batch['panel_1'], batch['panel_2'])

    def predict_step(self, batch, batch_idx):
        if self.per_view_normalization:
            batch['values'] = self.normalize(batch['values'])

        context_size = batch['tokens'].shape[1]
        nonzero_cnt = (batch['tokens'] != self.PAD_TOKEN_ID).sum(dim=1)
        # print(int(context_size), nonzero_cnt[0].item())

        batch = self.add_cls_token(batch)
        if self.debug and batch_idx % 20 == 0:
            print('batch tokens:', batch['tokens'][0])
            print('batch values:', batch['values'][0])
        
        padding_mask = (batch['tokens'] == self.PAD_TOKEN_ID) if self.mask_padding else torch.zeros_like(batch['tokens'], dtype=torch.bool, device=self.device)
        pred, embs, cell_embs = self(batch['tokens'], batch['values'], src_key_padding_mask=padding_mask)
        
        embs_mean = [embs[i, ~padding_mask[i]].mean(dim=0) for i in range(len(embs))]
        embs_mean = torch.stack(embs_mean, dim=0)
        
        if self.projection_dim:
            cell_embs = self.projection(cell_embs)
        
        return {'pred': pred, 
                'cls_cell_emb': cell_embs.float(), 
                'mean_cell_emb': embs_mean.float(),
                'context_sizes': (int(context_size), nonzero_cnt[random.randint(0, len(nonzero_cnt)-1)].item())
                }

    def on_train_epoch_end(self):
        if self.global_rank == 0:
            max_size = 1000
            columns = ["panel_name", "panel_size_1", "panel_size_2", "context_size_1", "context_size_2", "nonzero_size_1", "nonzero_size_2", 
                        "min_1", "min_2", "max_1", "max_2", "panel_intersect", "token_intersect"]
            try:
                table = wandb.Table(columns=columns, data=self.context_sizes['train'][:max_size])
                self.logger.experiment.log({"train/sample_stats": table})
            except:
                pass
        self.context_sizes['train'] = []
    
    def on_validation_epoch_end(self):     
        for val_name in self.val_loader_names:
            prefix = prefix=f'val/{val_name}' if val_name != 'same' else 'val'
            if self.global_rank == 0:
                max_size = 1000
                columns = ["panel_name", "panel_size_1", "panel_size_2", "context_size_1", "context_size_2", "nonzero_size_1", "nonzero_size_2", 
                            "min_1", "min_2", "max_1", "max_2", "panel_intersect", "token_intersect"]
                try:
                    table = wandb.Table(columns=columns, data=self.context_sizes['val'][val_name][:max_size])
                    self.logger.experiment.log({f"{prefix}/sample_stats": table})
                except:
                    pass
            self.context_sizes['val'][val_name] = []

    def _validate_panels(self, panel_1, panel_2):
        panel_1 = torch.cat(all_gather(panel_1), dim=0) if self.world_size > 1 else panel_1
        panel_2 = torch.cat(all_gather(panel_2), dim=0) if self.world_size > 1 else panel_2
        assert (panel_1[:, :5] == panel_1[0, :5]).all()
        assert (panel_2[:, :5] == panel_2[0, :5]).all()
        assert (panel_2 == self.PAD_TOKEN_ID).sum() == 0
        # assert np.intersect1d(panel_1[0].cpu(), panel_2[0].cpu()).size == 0


    def normalize(self, values, epsilon=1e-3):
        values = values.float() / (values.sum(dim=1, keepdim=True) + epsilon) * values.shape[1]
        values = torch.log1p(values)
        return values
    
    def split_inputs(self, batch, random_split=False, min_size=1):
        batch_1 = {}
        batch_2 = {}
        num_genes = batch['tokens'].shape[1]
        if random_split:
            context_size_1 = random.randint(min_size, num_genes - min_size)
            context_size_2 = random.randint(min_size, num_genes - context_size_1)
        else:
            context_size_1 = num_genes//2
            context_size_2 = num_genes//2
        for k, v in batch.items():
            if len(v.shape) > 1:
                batch_1[k], batch_2[k], _ = torch.split(v, [context_size_1, context_size_2, num_genes - context_size_1 - context_size_2], dim=1)
            else:
                batch_1[k], batch_2[k] = v, v
        return batch_1, batch_2
    

    def add_cls_token(self, batch):
        if isinstance(batch['tokens'], torch.Tensor):
            batch['tokens'] = torch.cat([
                torch.full(
                    (batch['tokens'].shape[0], 1),
                    self.CLS_TOKEN_ID,
                    dtype=batch['tokens'].dtype,
                    device=batch['tokens'].device,
                ), batch['tokens']], dim=1)

            batch['values'] = torch.cat([
                torch.full(
                    (batch['values'].shape[0], 1),
                    self.CLS_VALUE,
                    dtype=batch['values'].dtype,
                    device=batch['values'].device,
                ), batch['values']], dim=1)
        else:
            tokens, values = batch['tokens'], batch['values']
            cls_token_id = torch.tensor([self.CLS_TOKEN_ID], device=tokens[0].device, dtype=tokens[0].dtype)
            cls_token_val = torch.tensor([self.CLS_VALUE], device=values[0].device, dtype=values[0].dtype)
            batch['tokens'] = [torch.cat([cls_token_id, t], dim=0) for t in tokens]
            batch['values'] = [torch.cat([cls_token_val, v], dim=0) for v in values]
        return batch

    # balanced neg/pos sampling + binary cross entropy
    def _contrastive_binary_loss(self, logits):
        batch_size = len(logits)
        same_cell_pred = F.sigmoid(logits)

        neg_pairs = list(combinations(range(batch_size), 2))
        neg_pairs = random.sample(neg_pairs, batch_size)
        pos_pairs = [(i, i) for i in range(batch_size)]

        cont_target = torch.cat(
            [torch.zeros(len(neg_pairs)), torch.ones(len(pos_pairs))]).to(self.device)
        cont_pred = torch.cat(
            [same_cell_pred[list(zip(*neg_pairs))], same_cell_pred[list(zip(*pos_pairs))]])
        loss_cont = F.binary_cross_entropy(cont_pred, cont_target)
        acc_cont = self.binarcy_accuracy(cont_pred, cont_target)
        return loss_cont, acc_cont

    def _clip_loss(self, logits):
        loss_1, acc_cont_1, top5_acc_cont_1  = self._contrastive_multiclass_loss(logits)
        loss_2, acc_cont_2, top5_acc_cont_2 = self._contrastive_multiclass_loss(logits.t())
        return (loss_1 + loss_2) / 2.0, (acc_cont_1 + acc_cont_2) / 2.0, (top5_acc_cont_1 + top5_acc_cont_2) / 2.0

    def _contrastive_multiclass_loss(self, logits):
        cont_target = torch.arange(len(logits), device=self.device)
        
        loss_cont = F.cross_entropy(logits, cont_target)
        acc_cont = (logits.argmax(dim=1) == cont_target).float().mean()
        
        if len(logits) >= 5:
            top5_acc_cont = (logits.topk(5, dim=1)[1] == cont_target.unsqueeze(1)).any(dim=1).float().mean()
        else:
            top5_acc_cont = 0.0
        
        return loss_cont, acc_cont, top5_acc_cont
    

    def _knn_accuracy(self, logits, labels_1, labels_2, k=5):
        logits_diag = logits - torch.eye(len(logits), device=self.device) * self.logit_scale.exp() * 2

        neighbors = logits_diag.topk(k, dim=1)[1]
        neighbor_labels = labels_2[neighbors]
        label_acc = (torch.mode(neighbor_labels, dim=1)[0] == labels_1).float().mean()
        return label_acc

    def _same_batch_score(self, logits, k=1):
        logits_diag = logits - torch.eye(len(logits), device=self.device) * self.logit_scale.exp() * 2
        labels_1 = torch.cat([torch.zeros(len(logits)//2, device=self.device), torch.ones(len(logits)//2, device=self.device)], dim=0)
        labels_2 = torch.cat([torch.ones(len(logits)//2, device=self.device), torch.zeros(len(logits)//2, device=self.device)], dim=0)

        neighbors = logits_diag.topk(k, dim=1)[1]
        neighbor_labels = labels_2[neighbors]
        same_batch_score = (neighbor_labels == labels_1.unsqueeze(1)).float().mean()
        return same_batch_score
    
    def _knn_r2(self, logits, labels_1, labels_2, k=5, ignore_self=False):
        if ignore_self:
            logits = logits - torch.eye(len(logits), device=self.device) * self.logit_scale.exp() * 2

        neighbors = logits.topk(k, dim=1)[1]
        neighbor_labels = labels_2[neighbors]
        preds = neighbor_labels.float().mean(dim=1)
        # rmse = torch.sqrt(((preds - labels_1.float())**2).mean())
        r2 = r2_score(preds, labels_1.float())
        return r2

# Cosine Scheduler for training
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_steps, min_lr=0.0):
        self.warmup = warmup
        self.max_num_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [max(self.min_lr, base_lr * lr_factor) for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_steps))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    
class WarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup):
        self.warmup = warmup
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 1.0
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor