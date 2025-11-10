import os
import torch
from omegaconf import OmegaConf, DictConfig
from lamin_dataloader.dataset import GeneIdTokenizer
from concept.data.datamodules import AnnDataModule
from concept.model import BiEncoderContrastiveModel
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse
import lightning as L
import pandas as pd
from lightning.pytorch.callbacks import BasePredictionWriter
import os

class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        mean_cell_embs = torch.cat([item['mean_cell_emb'] for item in predictions], dim=0).cpu().detach().numpy()
        cls_cell_embs = torch.cat([item['cls_cell_emb'] for item in predictions], dim=0).cpu().detach().numpy()
        context_sizes = [item['context_sizes'] for item in predictions]
        
        batch_indices = np.concatenate(batch_indices[0])
        assert np.all(batch_indices[:-1] <= batch_indices[1:]), "batch_indices is not sorted"
        np.save(self.output_dir / f'cell_embs_cls.npy', cls_cell_embs)
        np.save(self.output_dir / f'cell_embs_mean.npy', mean_cell_embs)
        np.save(self.output_dir / f'context_sizes.npy', context_sizes)
        np.save(self.output_dir / f'batch_indices.npy', batch_indices)



def get_embs(cfg: DictConfig, ckpt_path: str, emb_path: str):
    
    gene_mapping = pd.read_pickle(cfg.PATH.gene_mapping_path).to_dict()
    
    model_args = {
        'config': cfg.model,
        'pad_token_id': gene_mapping['<pad>'],
        'cls_token_id': gene_mapping['<cls>'],
        'vocab_size': len(gene_mapping),
    }
    model = BiEncoderContrastiveModel.load_from_checkpoint(ckpt_path, **model_args)
    
    split = {}
    for key, filenames in cfg.PATH.SPLIT.items():
        split[key] = [os.path.join(cfg.PATH.ADATA_PATH, file) for file in filenames]

    datamodule_args = {
        'split': split,
        'panels_path': cfg.PATH.PANELS_PATH,
        'columns': cfg.datamodule.columns,
        'normalization': cfg.datamodule.normalization,
        'gene_sampling_strategy': cfg.datamodule.gene_sampling_strategy,
        'dataset_kwargs': {**cfg.datamodule.dataset},
        'dataloader_kwargs': {**cfg.datamodule.dataloader},
        'tokenizer': GeneIdTokenizer(gene_mapping)
    }
    datamodule = AnnDataModule(**datamodule_args)

    
    trainer_kwargs = {
        'accelerator': cfg.model.training.accelerator,
        'devices': cfg.model.training.devices,
        'callbacks': [PredictionWriter(output_dir=emb_path, write_interval="epoch")]
    }
    trainer = L.Trainer(**trainer_kwargs, strategy="ddp", precision='bf16-mixed')


    with torch.no_grad():
        model.eval()
        results = trainer.predict(model, datamodule.test_dataloader())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, help="Wandb-id of the run", required=True)
    parser.add_argument("--checkpoint", type=str, default='min_val_loss.ckpt', help="checkpoint name for loading")
    parser.add_argument("--max_tokens", type=int, default=None, help="number of tokens to use")
    parser.add_argument("--batch_size", type=int, default=None, help="batch size to use")
    parser.add_argument("--devices", type=int, default=-1, help="devices to use")
    parser.add_argument("--dataset", type=str, default=None, required=True, help="path to the dataset")
    parser.add_argument("--data_dir", type=str, default='h5ads', help="adata filename to use")
    parser.add_argument("--filename", type=str, default='adata_0.h5ad', help="adata filename to use")
    parser.add_argument("--gene_sampling_strategy", type=str, default=None, help="gene sampling strategy to use")
    parser.add_argument("--entity", type=str, default='theislab-transformer', help="entity to use")
    parser.add_argument("--project", type=str, default='contrastive-transformer', help="project to use")
    args = parser.parse_args()

    wandb.login()
    api = wandb.Api()
    
    run = api.run(f'{args.entity}/{args.project}/{args.run_id}')
    print(f"Getting {args.data_dir}/{args.filename} embeddings for run {run.id} ... with gene sampling strategy {args.gene_sampling_strategy}")
    cfg = DictConfig(run.config)
    
    gpu_info = torch.cuda.get_device_properties(0)
    print(f"GPU Type: {gpu_info.name}")
    
    # Compatibility changes:
    if 'per_view_normalization' not in cfg.model:
        cfg.model.per_view_normalization = False
    if 'projection_dim' not in cfg.model:
        cfg.model.projection_dim = None
    if 'weight_decay' not in cfg.model.training:
        cfg.model.training.weight_decay = 0.0
    if 'min_lr' not in cfg.model.training:
        cfg.model.training.min_lr = 0.0
    if 'data_loading_speed_sanity_check' not in cfg.model:
        cfg.model.data_loading_speed_sanity_check = False
    if 'decoder_head' not in cfg.model:
        cfg.model.decoder_head = True
    if 'gene_sampling_strategy' in cfg.datamodule.dataset.train:
        cfg.datamodule.gene_sampling_strategy = cfg.datamodule.dataset.train.gene_sampling_strategy
    if 'gene_sampling_strategy' not in cfg.datamodule:
        cfg.datamodule.gene_sampling_strategy = 'top-nonzero'
    if 'model_speed_sanity_check' not in cfg.datamodule:
        cfg.datamodule.model_speed_sanity_check = False
    if 'min_tokens' not in cfg.model:
        cfg.model.min_tokens = None
    if 'max_tokens' not in cfg.model:
        cfg.model.max_tokens = None
    if 'mask_padding' not in cfg.model:
        cfg.model.mask_padding = False
    if 'flash_attention' not in cfg.model:
        cfg.model.flash_attention = False
    if 'pe_max_len' not in cfg.model:
        cfg.model.pe_max_len = 5000
    if 'loss_switch_step' not in cfg.model:
        cfg.model.loss_switch_step = 2000

    # change configs to match the test settings    
    if args.devices is not None:
        cfg.model.training.devices = args.devices
    
    max_tokens = args.max_tokens if args.max_tokens is not None else cfg.datamodule.dataset.train.max_tokens
    
    cfg.datamodule.dataset = {'test': {'max_tokens': max_tokens, 'variable_size': False, 'panel_size_min': None, 'panel_size_max': None, 'panel_selection': None}}
    cfg.datamodule.dataloader = {'test': {'shuffle': False, 'drop_last': False, 'batch_size': args.batch_size}}
    cfg.datamodule.columns = []
    
    
    cfg.PATH.DATASET_PATH = Path(cfg.PATH.PROJECT_DATA_PATH) / args.dataset
    cfg.PATH.ADATA_PATH = cfg.PATH.DATASET_PATH / args.data_dir
    cfg.PATH.SPLIT = {'test': [f'{args.filename}']}

    emb_path = Path(cfg.PATH.DATASET_PATH) / 'embs' / args.run_id / f'{cfg.datamodule.dataset["test"]["max_tokens"]}'/ args.checkpoint / args.filename
    os.makedirs(emb_path, exist_ok=True)
    
    if (emb_path / 'cell_embs_rank_0.npz').exists() or (emb_path / 'cell_embs_cls.npy').exists():
        print(f"Embeddings already exist in {emb_path} ...")
        exit(0)
    
    ckpt_path = os.path.join(cfg.PATH.CHECKPOINT_ROOT, args.run_id, args.checkpoint)
    
    get_embs(cfg, ckpt_path, emb_path)
    
    
# Usage example: refer to https://github.com/theislab/scConcept-reproducibility/blob/main/ct_rep/get_embs/get_embs.sh