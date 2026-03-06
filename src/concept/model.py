import logging
import math
import os
import random
from collections import defaultdict
from functools import partial
from itertools import combinations
from typing import List, Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from torch import Tensor, nn, optim
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.functional.regression import r2_score

from .modules.bert_padding import pad_input, unpad_input
from .modules.flash_attention_layer import FlashTransformerEncoderLayer
from .modules.transformer import TransformerEncoder
from .tensor_gather import GatherLayer

logger = logging.getLogger(__name__)


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
        self.flash_attention = config["flash_attention"]
        self.dim_gene_embs = config.get("dim_gene_embs", config["dim_model"])
        self.dim_model = config["dim_model"]
        self.dim_hid = config["dim_hid"]
        self.num_head = config["num_head"]
        self.nlayers = config["nlayers"]
        self.dropout = config["dropout"]
        self.decoder_head = config["decoder_head"]
        self.input_encoding = config["input_encoding"]
        self.PAD_TOKEN_ID = pad_token_id
        self.CLS_TOKEN_ID = cls_token_id
        self.masking_rate = config["training"]["masking_rate"]
        self.lr = config["training"]["lr"]
        self.weight_decay = config["training"]["weight_decay"]
        self.optimizer_class = config["training"]["optimizer_class"]
        self.scheduler = config["training"]["scheduler"]
        self.warmup = config["training"]["warmup"]
        self.max_steps = config["training"]["max_steps"]
        self.min_lr = config["training"]["min_lr"]
        self.log_every_n_steps = config["training"].get("log_every_n_steps", 100)
        self.values_only_sanity_check = config["values_only_sanity_check"]
        self.data_loading_speed_sanity_check = config["data_loading_speed_sanity_check"]
        self.norm_scheme = config.get("norm_scheme", "post")
        self.activation = config.get("activation", "relu")

        encoder_layers = FlashTransformerEncoderLayer(
            self.dim_model,
            self.num_head,
            self.dim_hid,
            self.dropout,
            batch_first=True,
            use_flash_attn=self.flash_attention,
            norm_scheme=self.norm_scheme,
            activation=self.activation,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)
        # self.transformer_encoder = torch.compile(self.transformer_encoder) #todo: check compilation

        if self.decoder_head:
            self.expression_decoder = GeneExpressionDecoder(self.dim_model)

        self.cls_embedding = nn.Parameter(torch.zeros(self.dim_model))

    def _encode(
        self,
        tokens: Tensor,
        values: Tensor,
        seq_lengths: List[int] = None,
    ) -> Tensor:
        batch_size = tokens.size(0)

        src_key_padding_mask = tokens == self.PAD_TOKEN_ID

        gene_embs = self._encode_gene_tokens(tokens)

        # Prepend learnable CLS embedding at position 0
        cls_emb = self.cls_embedding.view(1, 1, -1).expand(batch_size, 1, -1)
        gene_embs = torch.cat([cls_emb, gene_embs], dim=1)

        # Prepend a CLS value placeholder so value/rank encoders stay aligned
        cls_val = torch.full((batch_size, 1), self.CLS_VALUE, dtype=values.dtype, device=values.device)
        values = torch.cat([cls_val, values], dim=1)

        # CLS position is never padding
        cls_col = torch.zeros(batch_size, 1, dtype=torch.bool, device=gene_embs.device)
        src_key_padding_mask = torch.cat([cls_col, src_key_padding_mask], dim=1)

        if self.flash_attention:
            assert seq_lengths is not None, "seq_lengths must be provided for flash attention"
            max_length = tokens.size(1) + 1  # +1 for CLS
            seq_lengths = [l + 1 for l in seq_lengths]

            gene_embs, indices, cu_seqlens, max_seqlen, seqlens = unpad_input(
                gene_embs, ~src_key_padding_mask, seq_lengths=seq_lengths
            )

            if self.input_encoding == "rank_encoding":
                # total_embs = self.positional_encoder(gene_embs, seqlens=list(seqlens))
                # Faster implementation:
                pe = self.positional_encoder.pe[:, :max_length, :]
                pe = pe.repeat(batch_size, 1, 1)
                pe, _, _, _, _ = unpad_input(pe, ~src_key_padding_mask, seq_lengths=seq_lengths)
                total_embs = gene_embs + pe
            elif self.input_encoding == "value_encoding":
                values, _, _, _, _ = unpad_input(values.unsqueeze(-1), ~src_key_padding_mask, seq_lengths=seq_lengths)
                values = values.squeeze(-1)
                value_embs = self._encode_values(values)
                total_embs = gene_embs + value_embs

            embs_jagged = self.transformer_encoder(total_embs, cu_seqlens=cu_seqlens, max_seqlen=max_length)
            embs_padded = pad_input(embs_jagged, indices, batch_size, max_length)
            cell_embs = embs_padded[:, 0, :]
            # cell_embs_jagged = embs_jagged[cu_seqlens[:-1]]
            # assert torch.equal(cell_embs_jagged, cell_embs), "cell_embs_jagged and cell_embs are not the same"
        else:
            if self.input_encoding == "rank_encoding":
                total_embs = self.positional_encoder(gene_embs)
            else:
                value_embs = self._encode_values(values)
                total_embs = gene_embs + value_embs

            embs_padded = self.transformer_encoder(total_embs, key_padding_mask=src_key_padding_mask)
            cell_embs = embs_padded[:, 0, :]

        return embs_padded, cell_embs

    def _encode_gene_tokens(self, tokens: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement _encode_gene_tokens method.")

    def _encode_values(self, values: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement _encode_values method.")

    def forward(self, input_tokens, input_values, seq_lengths: List[int] = None):
        embs_padded, cell_embs = self._encode(input_tokens, input_values, seq_lengths=seq_lengths)
        pred = self.expression_decoder(embs_padded) if self.decoder_head else None

        return pred, embs_padded, cell_embs

    def _step(self, batch):
        raise NotImplementedError("Subclasses must implement _step method.")

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train/loss", loss, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val/loss", loss, sync_dist=False)

    def _get_scheduler(self, optimizer):
        if self.scheduler == "warmup":
            lr_scheduler = WarmupScheduler(optimizer, warmup=self.warmup)
        elif self.scheduler == "warmup_cosine":
            lr_scheduler = CosineWarmupScheduler(
                optimizer, warmup=self.warmup, max_steps=self.max_steps, min_lr=self.min_lr
            )
        return lr_scheduler

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)  # default Adam
        if self.optimizer_class == "AdamW":
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.scheduler:
            lr_scheduler = self._get_scheduler(optimizer)
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
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


class GeneEncoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        dim_gene_embs: int,
        dim_model: int,
        padding_idx: Optional[int] = None,
        pretrained_vocabulary: Optional[Tensor] = None,
        freeze_pretrained_vocabulary: bool = None,
    ):
        super().__init__()
        self.dim_gene_embs = dim_gene_embs
        self.dim_model = dim_model
        self.freeze_pretrained_vocabulary = freeze_pretrained_vocabulary
        self.pretrained_vocabulary_available = pretrained_vocabulary is not None

        # If a pretrained vocabulary is provided, build custom embedding weights
        if self.pretrained_vocabulary_available:
            assert pretrained_vocabulary.shape[0] == n_genes, (
                f"Pretrained vocabulary must have the same length as the number of genes: {pretrained_vocabulary.shape[0]} == {n_genes}"
            )
            assert freeze_pretrained_vocabulary is not None, (
                "freeze_pretrained_vocabulary must be provided if pretrained_vocabulary is provided"
            )

            pretrained_dim = pretrained_vocabulary.shape[1]
            logger.info(
                f"Using pretrained embeddings for {len(pretrained_vocabulary)} genes of size {pretrained_dim} with dim_gene_embs={dim_gene_embs}"
            )

            self.pretrained_embs = nn.Embedding.from_pretrained(
                pretrained_vocabulary, freeze=freeze_pretrained_vocabulary, padding_idx=padding_idx
            )
            self.adapter1 = nn.Linear(pretrained_dim, dim_gene_embs, bias=True)

            self.embedding = nn.Embedding(
                n_genes,
                dim_gene_embs,
                padding_idx=padding_idx,
                _weight=torch.zeros(n_genes, dim_gene_embs, dtype=torch.float),
            )
        else:
            self.embedding = nn.Embedding(n_genes, dim_gene_embs, padding_idx=padding_idx)

        self.adapter2 = nn.Linear(dim_gene_embs, dim_model, bias=True)
        self.enc_norm = nn.LayerNorm(dim_model)

    def forward(self, x: Tensor, add_learnable_embs: bool = False) -> Tensor:
        x_learnable = self.embedding(x)
        if self.pretrained_vocabulary_available:
            x_pretrained = self.pretrained_embs(x)
            x_pretrained = self.adapter1(x_pretrained)

        if self.pretrained_vocabulary_available:
            x = x_pretrained + x_learnable * int(add_learnable_embs)
        else:
            x = x_learnable

        x = self.adapter2(x)
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
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # Shape: (d_model/2,)

        # Sinusoidal function for even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Cosine function for odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)  # Register as a buffer (non-trainable parameter)

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


class ContrastiveModel(BaseTransformerModel):
    def __init__(
        self,
        config,
        pad_token_id: int,
        cls_token_id: int,
        vocab_size: int,
        pretrained_vocabulary: Optional[dict] = None,
        precomp_embs_key: str = None,
        world_size: int = 1,
        val_loader_names=[],
        obs_keys: List[str] = [],
        debug: bool = False,
    ):
        if config["mlm_loss_weight"] > 0:
            assert config["decoder_head"] == True, "Decoder head must be enabled for MLM loss"

        super().__init__(config, pad_token_id, cls_token_id, debug=debug)

        self.mlm_loss_weight = config["mlm_loss_weight"]
        self.cont_loss_weight = config["cont_loss_weight"]
        self.contrastive_loss = config["contrastive_loss"]
        self.loss_switch_step = config["loss_switch_step"]
        self.logit_scale_init_value = config["logit_scale_init_value"]
        self.projection_dim = config["projection_dim"]
        self.pe_max_len = config["pe_max_len"]
        self.precomp_embs_key = precomp_embs_key
        self.vocab_size = vocab_size
        self.world_size = world_size
        self.val_loader_names = val_loader_names
        self.obs_keys = list(obs_keys)
        assert self.contrastive_loss in ["binary", "multiclass"]
        self.LOGGING_STEP = False

        self.use_learnable_embs_freq = config["training"]["use_learnable_embs_freq"]
        self.gene_token_encoder = GeneEncoder(
            self.vocab_size,
            self.dim_gene_embs,
            self.dim_model,
            padding_idx=pad_token_id,
            pretrained_vocabulary=pretrained_vocabulary,
            freeze_pretrained_vocabulary=config["training"]["freeze_pretrained_vocabulary"],
        )
        if self.input_encoding == "value_encoding":
            self.value_encoder = ContinuousValueEncoder(self.dim_model, dropout=0.0)
        elif self.input_encoding == "rank_encoding":
            self.positional_encoder = PositionalEncoding(self.dim_model, max_len=self.pe_max_len)

        self.binarcy_accuracy = BinaryAccuracy()
        self.logit_scale = nn.Parameter(torch.tensor(float(self.logit_scale_init_value)), requires_grad=True)
        if self.projection_dim:
            self.projection = nn.Linear(self.dim_model, self.projection_dim, bias=False)

        self.sample_stats = {"train": [], "val": defaultdict(list)}
        self.logit_masks = {}
        self.stage = None

    def log_metric(self, metric_name, value, **kwargs):
        kwargs = {"add_dataloader_idx": False, "sync_dist": True, **kwargs}
        if self.stage == "train":
            self.log(metric_name, value, on_step=True, on_epoch=False, **kwargs)
        self.log(metric_name + "_epoch", value, on_step=False, on_epoch=True, **kwargs)

    def log_metrics_dict(self, metrics: dict, prefix: str, batch_size=None):
        kwargs = {} if batch_size is None else {"batch_size": batch_size}
        for name, value in metrics.items():
            self.log_metric(f"{prefix}/{name}", value, **kwargs)

    def _encode_gene_tokens(self, tokens: Tensor) -> Tensor:
        if self.stage is None:
            logger.warning("model.stage is not set! setting model.stage to 'predict'")
            self.stage = "predict"

        # Deterministically add learnable_specific_embs based on use_learnable_embs_freq
        add_learnable_embs = False
        if self.gene_token_encoder.pretrained_vocabulary_available and self.use_learnable_embs_freq is not None:
            if self.stage == "train" and not self.LOGGING_STEP:
                add_learnable_embs = int((self.global_step + 1) * self.use_learnable_embs_freq) > int(
                    self.global_step * self.use_learnable_embs_freq
                )
            else:
                add_learnable_embs = True

        return self.gene_token_encoder(tokens, add_learnable_embs=add_learnable_embs)

    def _encode_values(self, values: Tensor) -> Tensor:
        return self.value_encoder(values)

    def configure_optimizers(self):
        # Separate parameters into those with and without weight decay
        # Frozen embeddings should not have weight decay applied
        if self.gene_token_encoder.pretrained_vocabulary_available:
            params_with_decay = []
            params_without_decay = []

            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if hasattr(self, "gene_token_encoder") and param is self.gene_token_encoder.embedding.weight:
                    params_without_decay.append(param)
                else:
                    params_with_decay.append(param)

            param_groups = [
                {"params": params_with_decay, "weight_decay": self.weight_decay},
                {"params": params_without_decay, "weight_decay": 0.0},
            ]
        else:
            param_groups = [{"params": self.parameters(), "weight_decay": self.weight_decay}]

        optimizer = optim.Adam(param_groups, lr=self.lr)  # default Adam
        if self.optimizer_class == "AdamW":
            optimizer = optim.AdamW(param_groups, lr=self.lr)

        if self.scheduler:
            lr_scheduler = self._get_scheduler(optimizer)
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
        else:
            return optimizer

    def _step(self, batch, batch_idx, log_prefix="train"):
        assert self.stage in ["train", "val"], f"Invalid stage: {self.stage}"
        metrics = {}

        batch_1 = {
            "tokens": batch["tokens_1"],
            "values": batch["values_1"],
            "panel": batch["panel_1"],
            "panel_name": batch["panel_name_1"],
            "seq_lengths": batch["seq_length_1"],
        }
        batch_2 = {
            "tokens": batch["tokens_2"],
            "values": batch["values_2"],
            "panel": batch["panel_2"],
            "panel_name": batch["panel_name_2"],
            "seq_lengths": batch["seq_length_2"],
        }

        if self.debug and batch_idx < 5 and self.stage == "train":
            logger.debug(f"batch_1 values: {batch_1['values'][0]}")
            logger.debug(f"batch_2 values: {batch_2['values'][0]}")

        if self.values_only_sanity_check:
            batch_1["values"] = batch_1["values"][:, torch.randperm(batch_1["tokens"].size(1))]
            batch_2["values"] = batch_2["values"][:, torch.randperm(batch_2["tokens"].size(1))]

        if self.masking_rate > 0:
            batch_1["values_masked"], batch_1["masked_positions"] = self.mask_values(
                batch_1["values"], self.masking_rate
            )
            batch_2["values_masked"], batch_2["masked_positions"] = self.mask_values(
                batch_2["values"], self.masking_rate
            )
        else:
            batch_1["values_masked"] = batch_1["values"]
            batch_2["values_masked"] = batch_2["values"]

        # Forward both views for maximizing GPU utilization instead of calling forward twice!
        pred_1, pred_2, cell_embs_1, cell_embs_2 = self.forward_pair(
            batch_1["tokens"],
            batch_1["values_masked"],
            batch_2["tokens"],
            batch_2["values_masked"],
            batch_1["seq_lengths"],
            batch_2["seq_lengths"],
        )

        loss_mlm = 0.0
        if self.mlm_loss_weight > 0:
            loss_mlm = self._mlm_loss(pred_1, batch_1["values"], batch_1["masked_positions"]) + self._mlm_loss(
                pred_2, batch_2["values"], batch_2["masked_positions"]
            )

        cell_embs_1 = self.all_gather_concat(cell_embs_1)
        cell_embs_2 = self.all_gather_concat(cell_embs_2)

        if self.projection_dim:
            cell_embs_1 = self.projection(cell_embs_1)
            cell_embs_2 = self.projection(cell_embs_2)

        if not self.precomp_embs_key or self.precomp_embs_key not in batch:
            cell_embs_1 = F.normalize(cell_embs_1, p=2, dim=1)
            cell_embs_2 = F.normalize(cell_embs_2, p=2, dim=1)
            logits = torch.mm(cell_embs_1, cell_embs_2.t()) * self.logit_scale.exp()
            cell_embs_concat_1 = torch.concat([cell_embs_1, cell_embs_2], dim=0)
            cell_embs_concat_2 = torch.concat([cell_embs_2, cell_embs_1], dim=0)
            logits_both_batch = torch.mm(cell_embs_concat_1, cell_embs_concat_2.t()) * self.logit_scale.exp()
        else:
            cell_embs_2 = self.all_gather_concat(batch[self.precomp_embs_key])
            logits = (1.0 / (torch.cdist(cell_embs_1, cell_embs_2, p=2) + 1e-4)) * self.logit_scale.exp()
            cell_embs_concat_1 = torch.concat([cell_embs_1, cell_embs_2], dim=0)
            cell_embs_concat_2 = torch.concat([cell_embs_2, cell_embs_1], dim=0)
            logits_both_batch = (
                1.0 / (torch.cdist(cell_embs_concat_1, cell_embs_concat_2, p=2) + 1e-4)
            ) * self.logit_scale.exp()

        logit_size = len(logits_both_batch)
        logits_both_batch *= torch.roll(1 - torch.eye(logit_size, device=self.device), logit_size // 2, 1)

        items_mask = None
        items_mask_both_batch = None
        if "items_mask" in batch:
            items_mask = self.all_gather_concat(batch["items_mask"]).float()  # [N]
            logit_mask = torch.outer(items_mask, items_mask)  # [N, N]
            logits *= logit_mask
            items_mask_both_batch = torch.cat([items_mask, items_mask])  # [2N]
            logits_both_batch *= torch.outer(items_mask_both_batch, items_mask_both_batch)  # [2N, 2N]

        if self.contrastive_loss == "binary":
            loss_cont, recall_top1, recall_top5 = self._contrastive_binary_loss(logits)
            loss_cont_both_batch, recall_top1_both_batch, recall_top5_both_batch = self._contrastive_binary_loss(
                logits_both_batch
            )
        elif self.contrastive_loss == "multiclass":
            loss_cont, recall_top1, recall_top5 = self._clip_loss(logits, items_mask)
            loss_cont_both_batch, recall_top1_both_batch, recall_top5_both_batch = self._contrastive_multiclass_loss(
                logits_both_batch, items_mask_both_batch
            )

        if self.global_step < self.loss_switch_step:
            loss = self.mlm_loss_weight * loss_mlm + self.cont_loss_weight * loss_cont
        else:
            loss = self.mlm_loss_weight * loss_mlm + self.cont_loss_weight * loss_cont_both_batch

        metrics["loss"] = loss.detach()
        metrics["loss_cont"] = loss_cont.detach()
        if loss_mlm > 0:
            metrics["loss_mlm"] = loss_mlm.detach()
        metrics["recall@1"] = recall_top1.detach()
        metrics["recall@5"] = recall_top5.detach()
        metrics["recall@1_combined"] = recall_top1_both_batch.detach()
        metrics["recall@5_combined"] = recall_top5_both_batch.detach()

        ######################################################################
        # Log extra metrics for monitoring
        ######################################################################

        for key in self.obs_keys:
            if self.stage != "train" and key in batch:
                labels_1, labels_2 = batch[key], batch[key]
                labels_1 = self.all_gather_concat(labels_1)
                labels_2 = self.all_gather_concat(labels_2)
                label_acc = 0.5 * (
                    self._knn_accuracy(logits, labels_1, labels_2, items_mask) + self._knn_accuracy(logits.t(), labels_2, labels_1, items_mask)
                )
                metrics[f"knn_acc_{key}"] = label_acc.detach()

        if self.stage == "train" and self.LOGGING_STEP:
            global_rank = torch.tensor([self.global_rank] * len(batch_1["tokens"]), device=self.device)
            global_rank = self.all_gather_concat(global_rank)
            global_rank_acc = 0.5 * (
                self._knn_accuracy(logits, global_rank, global_rank, items_mask)
                + self._knn_accuracy(logits.t(), global_rank, global_rank, items_mask)
            )
            metrics["knn_acc_global_rank"] = global_rank_acc.detach()

        if self.stage != "train":
            views_mixing_score = self._views_mixing_score(logits_both_batch, items_mask_both_batch, k=1)
            views_mixing_score_top_5 = self._views_mixing_score(logits_both_batch, items_mask_both_batch, k=5)
            metrics["views_mixing_score"] = views_mixing_score.detach()
            metrics["views_mixing_score_top_5"] = views_mixing_score_top_5.detach()

        if self.debug and self.LOGGING_STEP:
            nonzero_cnt_1 = (batch_1["tokens"] != self.PAD_TOKEN_ID).sum(dim=1)
            nonzero_cnt_2 = (batch_2["tokens"] != self.PAD_TOKEN_ID).sum(dim=1)
            nonzero_cnt_1 = self.all_gather_concat(nonzero_cnt_1)
            nonzero_cnt_2 = self.all_gather_concat(nonzero_cnt_2)

            length_sim = torch.mm(nonzero_cnt_1.unsqueeze(1).float(), nonzero_cnt_2.unsqueeze(0).float())
            length_logit_corr = torch.tensor(
                [torch.corrcoef(torch.stack([logits[i], length_sim[i]]))[0, 1] for i in range(logits.size(0))]
            )
            length_logit_corr = torch.mean(length_logit_corr[~torch.isnan(length_logit_corr)]).item()
            metrics["corr_seqlength"] = length_logit_corr

            length_r2 = 0.5 * (
                self._knn_r2(logits, nonzero_cnt_1, nonzero_cnt_2)
                + self._knn_r2(logits.t(), nonzero_cnt_2, nonzero_cnt_1)
            )
            metrics["knn_r2_seqlength"] = length_r2.detach() if hasattr(length_r2, "detach") else length_r2

        if (
            self.debug
            and self.world_size == 1
            and self.global_rank == 0
            and self.stage == "train"
            and batch_idx % 1000 == 0
        ):
            logger.debug(f"Argmax: {logits_both_batch.argmax(dim=1)}")
            logger.debug(f'batch_1["tokens"][0]: {batch_1["tokens"][0]}')
            logger.debug(f'batch_1["values"][0]: {batch_1["values"][0]}')
            logger.debug(f'batch_2["tokens"][0]: {batch_2["tokens"][0]}')
            logger.debug(f'batch_2["values"][0]: {batch_2["values"][0]}')

            idx = int(logits_both_batch.argmax(dim=1)[0])
            if idx < logit_size // 2:
                logger.debug(f"match in batch_2, {idx} tokens: {batch_2['tokens'][idx]}")
                logger.debug(f"match in batch_2, {idx} values: {batch_2['values'][idx]}")
            else:
                idx = idx - logit_size // 2
                logger.debug(f"match in batch_1, {idx} tokens: {batch_1['tokens'][idx]}")
                logger.debug(f"match in batch_1, {idx} values: {batch_1['values'][idx]}")
        ######################################################################

        return loss, metrics

    def training_step(self, batch, batch_idx):
        self.stage = "train"
        self.LOGGING_STEP = batch_idx % self.log_every_n_steps == 0

        if self.data_loading_speed_sanity_check:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return loss

        loss, metrics = self._step(batch, batch_idx, log_prefix="train")
        if self.LOGGING_STEP:
            self.log_metrics_dict(metrics, "train", batch_size=len(batch["tokens_1"]))
            sample_stats = self._get_sample_stats(batch)
            sample_stats.update({k: metrics[k] for k in ["recall@1", "recall@1_combined"]})
            self.sample_stats["train"].append(sample_stats)

        if self.debug and "panel_1" in batch and "panel_2" in batch and self.LOGGING_STEP:
            self._validate_panels(batch["panel_1"], batch["panel_2"])

        return loss

    def on_fit_start(self):
        if self.data_loading_speed_sanity_check:
            self.requires_grad_(False)

    def backward(self, loss, *args, **kwargs):
        if self.data_loading_speed_sanity_check:
            return
        super().backward(loss, *args, **kwargs)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.stage = "val"
        self.LOGGING_STEP = batch_idx % self.log_every_n_steps == 0

        if self.data_loading_speed_sanity_check:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return loss

        val_name = self.val_loader_names[dataloader_idx]
        prefix = f"val/{val_name}"

        loss, metrics = self._step(batch, batch_idx, log_prefix=prefix)
        self.log_metrics_dict(metrics, prefix, batch_size=len(batch["tokens_1"]))

        sample_stats = self._get_sample_stats(batch)
        sample_stats.update({k: metrics[k] for k in ["recall@1", "recall@1_combined"]})
        self.sample_stats["val"][val_name].append(sample_stats)

        if self.debug and "panel_1" in batch and "panel_2" in batch and self.LOGGING_STEP:
            self._validate_panels(batch["panel_1"], batch["panel_2"])

    def predict_step(self, batch, batch_idx, use_learnable_embs: bool = True):
        self.stage = "predict"
        self.use_learnable_embs_freq = int(use_learnable_embs)

        context_size = batch["tokens"].shape[1]
        nonzero_cnt = (batch["tokens"] != self.PAD_TOKEN_ID).sum(dim=1)
        # logger.debug("%d, %d", int(context_size), nonzero_cnt[0].item())

        if self.debug and batch_idx % 20 == 0:
            logger.debug(f"batch tokens: {batch['tokens'][0]}")
            logger.debug(f"batch values: {batch['values'][0]}")

        padding_mask = batch["tokens"] == self.PAD_TOKEN_ID
        if "seq_lengths" not in batch or batch["seq_lengths"] is None:
            batch["seq_lengths"] = (~padding_mask).sum(dim=1).tolist()

        pred, embs, cell_embs = self(batch["tokens"], batch["values"], seq_lengths=batch["seq_lengths"])

        # embs has CLS at position 0 (added inside _encode); skip it and average over gene positions
        embs_mean = [embs[i, 1:][~padding_mask[i]].mean(dim=0) for i in range(len(embs))]
        embs_mean = torch.stack(embs_mean, dim=0)

        if self.projection_dim:
            cell_embs = self.projection(cell_embs)

        return {
            "pred": pred,
            "cls_cell_emb": cell_embs.float(),
            "mean_cell_emb": embs_mean.float(),
            "context_sizes": (int(context_size), nonzero_cnt[random.randint(0, len(nonzero_cnt) - 1)].item()),
        }

    def all_gather_concat(self, tensor: Tensor) -> Tensor:
        if self.world_size > 1:
            return torch.cat(GatherLayer.apply(tensor.contiguous()), dim=0)
        else:
            return tensor

    def _convert_stats_tensors_to_scalars(self, stats_list):
        converted_stats = []
        for stats in stats_list:
            converted = {}
            for key, value in stats.items():
                if isinstance(value, torch.Tensor):
                    converted[key] = value.cpu().item()
                else:
                    converted[key] = value
            converted_stats.append(converted)
        return converted_stats

    def on_train_epoch_end(self):
        if self.global_rank == 0:
            max_size = 1000
            try:
                stats_train = self._convert_stats_tensors_to_scalars(self.sample_stats["train"])
                df = pd.DataFrame(stats_train)
                if len(df) > max_size:
                    indices = np.random.choice(len(df), max_size, replace=False)
                    df = df.iloc[indices]

                table = wandb.Table(dataframe=df)
                self.logger.experiment.log({"train/sample_stats": table})
            except:
                pass
        self.sample_stats["train"] = []

    def on_validation_epoch_end(self):
        for val_name in self.val_loader_names:
            prefix = f"val/{val_name}"
            if self.global_rank == 0:
                max_size = 1000
                try:
                    stats_val = self._convert_stats_tensors_to_scalars(self.sample_stats["val"][val_name])
                    df = pd.DataFrame(stats_val)
                    if len(df) > max_size:
                        indices = np.random.choice(len(df), max_size, replace=False)
                        df = df.iloc[indices]

                    table = wandb.Table(dataframe=df)
                    self.logger.experiment.log({f"{prefix}/sample_stats": table})
                except:
                    pass
            self.sample_stats["val"][val_name] = []

    def on_before_optimizer_step(self, optimizer):
        if self.trainer.global_step % self.log_every_n_steps == 0:
            norms = {}
            for n, layer in enumerate(self.transformer_encoder.layers):
                for name, p in layer.named_parameters():
                    if p.grad is not None:
                        norms[f"grad_norm/layer_{n}_{name}"] = p.grad.data.norm(2)
                    norms[f"param_norm/layer_{n}_{name}"] = p.data.norm(2)
            self.log_dict(norms, on_step=True, on_epoch=False, sync_dist=True)

    def _validate_panels(self, panel_1, panel_2):
        panel_1 = self.all_gather_concat(panel_1)
        panel_2 = self.all_gather_concat(panel_2)
        assert (panel_1[:, :5] == panel_1[0, :5]).all()
        assert (panel_2[:, :5] == panel_2[0, :5]).all()
        assert (panel_2 == self.PAD_TOKEN_ID).sum() == 0
        # assert np.intersect1d(panel_1[0].cpu(), panel_2[0].cpu()).size == 0

    @torch.no_grad()
    def _get_sample_stats(self, batch):
        panel_size_1 = batch["panel_1"].shape[1]
        panel_size_2 = batch["panel_2"].shape[1]
        context_size_1 = len(batch["tokens_1"][0])
        context_size_2 = len(batch["tokens_2"][0])
        nonzero_cnt_1 = (batch["tokens_1"][0] != self.PAD_TOKEN_ID).sum().detach()
        nonzero_cnt_2 = (batch["tokens_2"][0] != self.PAD_TOKEN_ID).sum().detach()
        try:
            values_min_1, values_min_2 = batch["values_1"][0].min().detach(), batch["values_2"][0].min().detach()
            values_max_1, values_max_2 = batch["values_1"][0].max().detach(), batch["values_2"][0].max().detach()
        except:
            values_min_1, values_min_2 = 0, 0
            values_max_1, values_max_2 = 0, 0

        # panel_intersect = torch.isin(torch.unique(batch["panel_1"][0]), torch.unique(batch["panel_2"][0])).sum().detach()
        # token_intersect = torch.isin(torch.unique(batch["tokens_1"][0]), torch.unique(batch["tokens_2"][0])).sum().detach()
        # For some reason torch.isin is slow!
        panel_intersect = np.intersect1d(batch["panel_1"][0].cpu().numpy(), batch["panel_2"][0].cpu().numpy()).size
        token_intersect = np.intersect1d(batch["tokens_1"][0].cpu().numpy(), batch["tokens_2"][0].cpu().numpy()).size


        batch_size = self.all_gather_concat(batch["items_mask"]).sum().detach() if "items_mask" in batch else len(batch["tokens_1"])

        sample_stats = {
            "_organism": batch["_organism"][0],
            "_tissue": ", ".join(list(batch["_tissue"][0])),
            "panel_name_1": batch["panel_name_1"],
            "panel_name_2": batch["panel_name_2"],
            "panel_size_1": int(panel_size_1),
            "panel_size_2": int(panel_size_2),
            "context_size_1": int(context_size_1),
            "context_size_2": int(context_size_2),
            "nonzero_size_1": nonzero_cnt_1,
            "nonzero_size_2": nonzero_cnt_2,
            "values_min_1": values_min_1,
            "values_min_2": values_min_2,
            "values_max_1": values_max_1,
            "values_max_2": values_max_2,
            "panel_intersect": panel_intersect,
            "token_intersect": token_intersect,
            "seq_length_sum_1": sum(batch["seq_length_1"]),  # Already on cpu
            "seq_length_sum_2": sum(batch["seq_length_2"]),  # Already on cpu
            "seq_length_sum_all": sum(batch["seq_length_1"]) + sum(batch["seq_length_2"]),  # Already on cpu
            "batch_size": batch_size,
        }

        for key in self.obs_keys:
            if key in batch:
                value = self.all_gather_concat(batch[key])
                sample_stats[f"same_{key}"] = (value[0] == value).all().detach()

        return sample_stats

    def add_cls_token(self, batch):
        if isinstance(batch["tokens"], torch.Tensor):
            batch["tokens"] = torch.cat(
                [
                    torch.full(
                        (batch["tokens"].shape[0], 1),
                        self.CLS_TOKEN_ID,
                        dtype=batch["tokens"].dtype,
                        device=batch["tokens"].device,
                    ),
                    batch["tokens"],
                ],
                dim=1,
            )

            batch["values"] = torch.cat(
                [
                    torch.full(
                        (batch["values"].shape[0], 1),
                        self.CLS_VALUE,
                        dtype=batch["values"].dtype,
                        device=batch["values"].device,
                    ),
                    batch["values"],
                ],
                dim=1,
            )

            if "seq_lengths" in batch and batch["seq_lengths"] is not None:
                batch["seq_lengths"] = [l + 1 for l in batch["seq_lengths"]]
        else:
            tokens, values = batch["tokens"], batch["values"]
            cls_token_id = torch.tensor([self.CLS_TOKEN_ID], device=tokens[0].device, dtype=tokens[0].dtype)
            cls_token_val = torch.tensor([self.CLS_VALUE], device=values[0].device, dtype=values[0].dtype)
            batch["tokens"] = [torch.cat([cls_token_id, t], dim=0) for t in tokens]
            batch["values"] = [torch.cat([cls_token_val, v], dim=0) for v in values]
            if "seq_lengths" in batch and batch["seq_lengths"] is not None:
                batch["seq_lengths"] = [l + 1 for l in batch["seq_lengths"]]
        return batch

    def pad_tensor(self, tensor, target_length, pad_value):
        pad_size = target_length - tensor.shape[1]
        if pad_size > 0:
            return F.pad(tensor, (0, pad_size), value=pad_value)
        return tensor

    def forward_pair(self, tokens_1, values_1, tokens_2, values_2, seq_lengths_1, seq_lengths_2):
        # This function is used to forward both views for efficiency instead of calling forward twice
        # Pad the batches to the same sequence length before concatenation
        max_len = max(tokens_1.shape[1], tokens_2.shape[1])

        padded_tokens_1 = self.pad_tensor(tokens_1, max_len, self.PAD_TOKEN_ID)
        padded_tokens_2 = self.pad_tensor(tokens_2, max_len, self.PAD_TOKEN_ID)
        padded_values_1 = self.pad_tensor(values_1, max_len, 0)
        padded_values_2 = self.pad_tensor(values_2, max_len, 0)

        combined_tokens = torch.cat([padded_tokens_1, padded_tokens_2], dim=0)
        combined_values = torch.cat([padded_values_1, padded_values_2], dim=0)
        combined_seq_lengths = seq_lengths_1 + seq_lengths_2

        pred, _, cell_embs = self(
            combined_tokens,
            combined_values,
            seq_lengths=combined_seq_lengths,
        )
        cell_embs_1, cell_embs_2 = torch.chunk(cell_embs, 2, dim=0)
        if pred is not None:
            pred_1, pred_2 = torch.chunk(pred, 2, dim=0)
        else:
            pred_1, pred_2 = None, None

        return pred_1, pred_2, cell_embs_1, cell_embs_2

    # balanced neg/pos sampling + binary cross entropy
    def _contrastive_binary_loss(self, logits):
        batch_size = len(logits)
        same_cell_pred = F.sigmoid(logits)

        neg_pairs = list(combinations(range(batch_size), 2))
        neg_pairs = random.sample(neg_pairs, batch_size)
        pos_pairs = [(i, i) for i in range(batch_size)]

        cont_target = torch.cat([torch.zeros(len(neg_pairs)), torch.ones(len(pos_pairs))]).to(self.device)
        cont_pred = torch.cat([same_cell_pred[list(zip(*neg_pairs))], same_cell_pred[list(zip(*pos_pairs))]])
        loss_cont = F.binary_cross_entropy(cont_pred, cont_target)
        recall_top1 = self.binarcy_accuracy(cont_pred, cont_target)
        recall_top5 = 0
        return loss_cont, recall_top1, recall_top5

    def _clip_loss(self, logits, items_mask=None):
        loss_1, recall_top1_1, recall_top5_1 = self._contrastive_multiclass_loss(logits, items_mask)
        loss_2, recall_top1_2, recall_top5_2 = self._contrastive_multiclass_loss(logits.t(), items_mask)
        return (
            (loss_1 + loss_2) / 2.0,
            (recall_top1_1 + recall_top1_2) / 2.0,
            (recall_top5_1 + recall_top5_2) / 2.0,
        )

    def _contrastive_multiclass_loss(self, logits, items_mask=None):
        if items_mask is None:
            items_mask = torch.ones(len(logits), device=self.device, dtype=torch.bool)
        items_mask = items_mask.float()

        cont_target = torch.arange(len(logits), device=self.device)

        loss_cont = F.cross_entropy(logits, cont_target, reduction='none')
        loss_cont = (loss_cont * items_mask).sum() / items_mask.sum()

        recall_top1 = (logits.argmax(dim=1) == cont_target).float()
        recall_top1 = (recall_top1 * items_mask).sum() / items_mask.sum()

        if len(logits) >= 5:
            recall_top5 = (logits.topk(5, dim=1)[1] == cont_target.unsqueeze(1)).any(dim=1).float()
            recall_top5 = (recall_top5 * items_mask).sum() / items_mask.sum()
        else:
            recall_top5 = torch.tensor(0.0, device=logits.device)

        return loss_cont, recall_top1, recall_top5

    def _knn_accuracy(self, logits, labels_1, labels_2, items_mask=None, k=5):
        if items_mask is None:
            items_mask = torch.ones(len(logits), device=self.device, dtype=torch.bool)
        items_mask = items_mask.float()

        logits_diag = logits - torch.eye(len(logits), device=self.device) * self.logit_scale.exp() * 2

        neighbors = logits_diag.topk(k, dim=1)[1]
        neighbor_labels = labels_2[neighbors]
        label_acc = (torch.mode(neighbor_labels, dim=1)[0] == labels_1).float()
        label_acc = (label_acc * items_mask).sum() / items_mask.sum()
        return label_acc

    def _views_mixing_score(self, logits, items_mask=None, k=1):
        if items_mask is None:
            items_mask = torch.ones(len(logits), device=self.device, dtype=torch.bool)
        items_mask = items_mask.float()

        logits_diag = logits - torch.eye(len(logits), device=self.device) * self.logit_scale.exp() * 2
        labels_1 = torch.cat(
            [torch.zeros(len(logits) // 2, device=self.device), torch.ones(len(logits) // 2, device=self.device)], dim=0
        )
        labels_2 = torch.cat(
            [torch.ones(len(logits) // 2, device=self.device), torch.zeros(len(logits) // 2, device=self.device)], dim=0
        )

        neighbors = logits_diag.topk(k, dim=1)[1]
        neighbor_labels = labels_2[neighbors]
        mixsing_score = (neighbor_labels != labels_1.unsqueeze(1)).float().mean(dim=1)
        mixsing_score = (mixsing_score * items_mask).sum() / items_mask.sum()
        return mixsing_score

    def _knn_r2(self, logits, labels_1, labels_2, items_mask=None, k=5, ignore_self=False):
        if items_mask is None:
            items_mask = torch.ones(len(logits), device=self.device, dtype=torch.bool)
        items_mask = items_mask.float()

        if ignore_self:
            logits = logits - torch.eye(len(logits), device=self.device) * self.logit_scale.exp() * 2

        neighbors = logits.topk(k, dim=1)[1]
        neighbor_labels = labels_2[neighbors]
        preds = neighbor_labels.float().mean(dim=1)
        # rmse = torch.sqrt(((preds - labels_1.float())**2).mean())
        r2 = r2_score(preds, labels_1.float(), multioutput='raw_values')
        r2 = (r2 * items_mask).sum() / items_mask.sum()
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
