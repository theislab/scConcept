import os
import torch
from omegaconf import DictConfig
from concept import scConcept
import wandb
import numpy as np
from pathlib import Path
import argparse
import anndata as ad



def get_embs(cfg: DictConfig, ckpt_path: str, adata_path: str, gene_id_column: str = None,
             batch_size: int = 32, max_tokens: int = None, gene_sampling_strategy: str = None):
    """
    Extract embeddings using scConcept API extract_embeddings method.
    
    Args:
        cfg: Configuration dictionary
        ckpt_path: Path to model checkpoint
        adata_path: Path to AnnData file
        batch_size: Batch size for dataloader (if None, uses config default)
        max_tokens: Maximum number of tokens per cell (if None, uses config default)
        gene_sampling_strategy: Gene sampling strategy (if None, uses config default)
        gene_id_column: Column name in adata.var to use as gene IDs (default: None, uses index)
    """
    # Use scConcept API to load config and model
    concept = scConcept()
    concept.load_config_and_model(
        config=cfg,
        model_path=ckpt_path,
        gene_mapping_path=cfg.PATH.gene_mapping_path,
        panels_dir=cfg.PATH.PANELS_PATH
    )
    
    # Load AnnData
    print(f"Loading AnnData from {adata_path}...")
    adata = ad.read_h5ad(adata_path)
        
    # Extract embeddings using the API method
    print(f"Extracting embeddings with batch_size={batch_size}, max_tokens={max_tokens}, gene_sampling_strategy={gene_sampling_strategy}")
    result = concept.extract_embeddings(
        adata=adata,
        batch_size=batch_size,
        max_tokens=max_tokens,
        gene_sampling_strategy=gene_sampling_strategy,
        gene_id_column=gene_id_column
    )
    
    return result
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_entity", type=str, required=True, help="wandb entity to use")
    parser.add_argument("--wandb_project", type=str, required=True, help="wandb project to use")
    parser.add_argument("--run_id", type=str, help="Wandb-id of the run", required=True)
    parser.add_argument("--checkpoint", type=str, default='min_val_loss.ckpt', help="checkpoint name for loading")
    parser.add_argument("--adata_path", type=str, required=True, help="Path to the AnnData file (.h5ad)")
    parser.add_argument("--gene_id_column", type=str, default=None, help="Column name in adata.var to use as gene IDs (default: None, uses index)")
    parser.add_argument("--output_emb_path", type=str, required=True, help="Path to save embeddings")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size to use")
    parser.add_argument("--max_tokens", type=int, default=None, help="number of tokens to use")
    parser.add_argument("--gene_sampling_strategy", type=str, default=None, help="gene sampling strategy to use")
    args = parser.parse_args()

    wandb.login()
    api = wandb.Api()
    
    run = api.run(f'{args.wandb_entity}/{args.wandb_project}/{args.run_id}')
    print(f"Getting embeddings from {args.adata_path} for run {run.id} ... with gene sampling strategy {args.gene_sampling_strategy}")
    
    # Load config from wandb and apply compatibility changes using scConcept API
    cfg = DictConfig(run.config)
    
    gpu_info = torch.cuda.get_device_properties(0)
    print(f"GPU Type: {gpu_info.name}")
    
    adata_path_obj = Path(args.adata_path)
    
    output_emb_path = Path(args.output_emb_path)
    
    if (output_emb_path / 'cell_embs_cls.npy').exists():
        print(f"Embeddings already exist in {output_emb_path} ...")
        exit(0)
    
    ckpt_path = os.path.join(cfg.PATH.CHECKPOINT_ROOT, args.run_id, args.checkpoint)
    
    result = get_embs(
        cfg=cfg, 
        ckpt_path=ckpt_path, 
        adata_path=args.adata_path,
        gene_id_column=args.gene_id_column,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        gene_sampling_strategy=args.gene_sampling_strategy,
    )
    
    # Save embeddings
    print(f"Saving embeddings to {output_emb_path}...")
    os.makedirs(output_emb_path, exist_ok=True)
    np.save(output_emb_path / 'cell_embs_cls.npy', result['cls_cell_emb'])
    np.save(output_emb_path / 'cell_embs_mean.npy', result['mean_cell_emb'])
    
    if 'context_sizes' in result:
        np.save(output_emb_path / 'context_sizes.npy', result['context_sizes'])
    
    print(f"Embeddings saved successfully to {output_emb_path}")
    

# Example usage:
# python src/concept/get_embs.py \
#   --wandb_entity <wandb_entity> \
#   --wandb_project <wandb_project> \
#   --run_id <wandb_run_id> \
#   --checkpoint <checkpoint_name>.ckpt \
#   --adata_path <path_to_input_adata.h5ad> \
#   --output_emb_path <output_directory> \
#   --batch_size 64 \

