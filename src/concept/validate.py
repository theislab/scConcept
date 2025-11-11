import os

import lightning as L
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from lamin_dataloader.dataset import GeneIdTokenizer
from concept.data.datamodules import AnnDataModule
from model import BiEncoderContrastiveModel
import wandb
from lightning.pytorch.strategies import DDPStrategy
import argparse
from pathlib import Path



def validate(cfg: DictConfig, ckpt_path: str):
    
    val_loader_names = sorted(list(cfg.datamodule.dataset.val.keys()))
    gene_mapping = pd.read_pickle(cfg.PATH.gene_mapping_path).to_dict()
    
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
        'dataloader_kwargs': {**OmegaConf.to_container(cfg.datamodule.dataloader, resolve=True, throw_on_missing=True)},
        'val_loader_names': val_loader_names,
        'tokenizer': GeneIdTokenizer(gene_mapping)
    }
    datamodule = AnnDataModule(**datamodule_args)

    trainer_kwargs = {
        'accelerator': cfg.model.training.accelerator,
        'devices': cfg.model.training.devices,
    }
    trainer = L.Trainer(**trainer_kwargs, strategy=DDPStrategy(find_unused_parameters=True), precision='bf16-mixed', use_distributed_sampler=False)

    model_args = {
        'config': cfg.model,
        'pad_token_id': gene_mapping['<pad>'],
        'cls_token_id': gene_mapping['<cls>'],
        'vocab_size': len(gene_mapping),
        'world_size': trainer.world_size, 
        'val_loader_names': val_loader_names, 
    }    
    model = BiEncoderContrastiveModel.load_from_checkpoint(ckpt_path, **model_args)


    model.eval()
    results = trainer.validate(model=model, datamodule=datamodule)
    res = {}
    for r in results:
        res.update(r)
    results = pd.DataFrame.from_dict(res, orient='index', columns=['value'])
    print(results)
    # os.makedirs(os.path.join(cfg.PATH.DATASET_PATH, 'results'), exist_ok=True)
    # results.to_csv(os.path.join(cfg.PATH.DATASET_PATH, 'results', f'validation_{run_id}_{cfg.datamodule.gene_panel}_{cfg.datamodule.val_sub_sample_frac}.csv'))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, help="Wandb-id of the run", required=True)
    parser.add_argument("--checkpoint", type=str, default='min_val_loss.ckpt', help="checkpoint name for loading")
    parser.add_argument("--dataset", type=str, help="Path to the dataset", required=True)
    parser.add_argument("--data_dir", type=str, default='h5ads', help="adata filename to use")
    parser.add_argument("--filenames", type=str, default='adata.h5ad', help="adata filename to use")
    parser.add_argument("--max_tokens", type=int, default=None, help="number of tokens to use")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size to use")
    parser.add_argument("--within_group_sampling", default=None, help="Whether to use within group sampling")
    parser.add_argument("--variable_size", default=False, action='store_true', help="Whether to use variable size")
    parser.add_argument("--label_key", type=str, default=None, help="Label key to use")
    parser.add_argument("--num_samples", type=int, default=50000, help="Number of samples to use")
    parser.add_argument("--panel_selection", type=str, default=None, help="Panel selection to use")
    parser.add_argument("--panel_filter_regex", type=str, default=None, help="Panel filter regex to use")
    parser.add_argument("--panel_size_min", type=int, default=None, help="Panel size min to use")
    parser.add_argument("--panel_size_max", type=int, default=None, help="Panel size max to use")
    parser.add_argument("--panel_overlap", default=False, action='store_true', help="Whether to use panel overlap")
    parser.add_argument("--feature_max_drop_rate", type=float, default=None, help="Drop rate to use")
    parser.add_argument("--panel_max_drop_rate", type=float, default=None, help="Panel max drop rate to use")
    parser.add_argument("--gene_sampling_strategy", type=str, default=None, help="Gene sampling strategy to use")
    args = parser.parse_args()

    print(f'Dataset: {args.dataset}/{args.data_dir}, Model:{args.run_id}/{args.checkpoint}')
    filenames = [name.strip() for name in args.filenames.split(",")]
    print(filenames)
    wandb.login()
    api = wandb.Api()
    try:
        run = api.run(f'theislab-transformer/contrastive-transformer/{args.run_id}')
    except:
        run = api.run(f'mojtaba-bahrami/contrastive-transformer/{args.run_id}')
    
    cfg = DictConfig(run.config)
    
    if 'mask_padding' not in cfg.model:
        cfg.model.mask_padding = False
    if 'flash_attention' not in cfg.model:
        cfg.model.flash_attention = False
    if 'pe_max_len' not in cfg.model:
        cfg.model.pe_max_len = 5000

    
    # Adapt the config to the finetuning task
    cfg.PATH.DATASET_PATH = Path(cfg.PATH.PROJECT_DATA_PATH) / args.dataset
    cfg.PATH.ADATA_PATH = Path(cfg.PATH.DATASET_PATH) / args.data_dir
    # cfg.PATH.SPLIT = {'val': [f'{args.filename}']}
    cfg.PATH.SPLIT = {'val': filenames}

    max_tokens = args.max_tokens if args.max_tokens is not None else cfg.datamodule.dataset.train.max_tokens
    
    val_name = f'within_{args.panel_selection}_{args.panel_filter_regex}_{max_tokens}_B{args.batch_size}' if args.within_group_sampling else f'across_G{max_tokens}_B{args.batch_size}'
    if args.variable_size:
        val_name += '_variable'
    cfg.datamodule.dataset = {'val': {val_name: {'max_tokens': max_tokens, 'variable_size': args.variable_size, 
                                                 'panel_selection': args.panel_selection, 'panel_filter_regex': args.panel_filter_regex,
                                                 'panel_size_min': args.panel_size_min, 'panel_size_max': args.panel_size_max, 'panel_overlap': args.panel_overlap,
                                                 'feature_max_drop_rate': args.feature_max_drop_rate,
                                                 'panel_max_drop_rate': args.panel_max_drop_rate,
                                                 }
                                      }
                              }
    cfg.datamodule.dataloader = {'val': {val_name: {'within_group_sampling': args.within_group_sampling,
                                                    'batch_size': args.batch_size, 
                                                    'shuffle': True, 
                                                    'drop_last': True, 
                                                    'num_samples': args.num_samples,
                                                    'num_workers': 10,
                                                    }
                                         }
                                 }
    cfg.datamodule.columns = [args.label_key] if args.label_key is not None else []
    if args.gene_sampling_strategy is not None:
        cfg.datamodule.gene_sampling_strategy = args.gene_sampling_strategy 

    ckpt_path = os.path.join(cfg.PATH.CHECKPOINT_ROOT, args.run_id, args.checkpoint)

    validate(cfg, ckpt_path)



# Usage Example: refert to https://github.com/theislab/scConcept-reproducibility/blob/main/scripts/validate.sh