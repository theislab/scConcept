import os
import torch
from concept.scConcept import scConcept
import numpy as np
from pathlib import Path
import argparse


def get_embs(run_id: str, dataset: str, filename: str, max_tokens: int, batch_size: int,
             gene_sampling_strategy: str, checkpoint: str = 'min_val_loss.ckpt',
             data_dir: str = 'h5ads', entity: str = 'theislab-transformer', 
             project: str = 'contrastive-transformer'):
    """
    Generate embeddings using InMemoryTokenizedDataset.
    
    Args:
        run_id: Wandb run ID
        dataset: Dataset name/path
        filename: AnnData filename
        max_tokens: Maximum number of tokens per cell
        batch_size: Batch size for dataloader
        gene_sampling_strategy: Gene sampling strategy ('top-nonzero', etc.)
        checkpoint: Checkpoint name for loading
        data_dir: Data directory name
        entity: Wandb entity
        project: Wandb project
    """
    
    # Create scConcept instance
    concept = scConcept(entity=entity, project=project)
    
    # Load model
    concept.load_config_and_model(run_id=run_id, checkpoint=checkpoint)
    
    # Set paths
    dataset_path = Path(concept.cfg.PATH.PROJECT_DATA_PATH) / dataset
    adata_path = dataset_path / data_dir / filename
    
    # Create output directory
    emb_path = dataset_path / 'embs' / run_id / f'{max_tokens}' / checkpoint / filename
    
    # Check if embeddings already exist
    if (emb_path / 'cell_embs_cls.npy').exists():
        print(f"Embeddings already exist in {emb_path} ...")
        return
    
    # Extract embeddings
    result = concept.extract_embeddings_from_file(
        adata_path=str(adata_path),
        batch_size=batch_size,
        max_tokens=max_tokens,
        gene_sampling_strategy=gene_sampling_strategy
    )
    
    # Save embeddings
    print(f"Saving embeddings to {emb_path}...")
    os.makedirs(emb_path, exist_ok=True)
    np.save(emb_path / 'cell_embs_cls.npy', result['cls_cell_emb'])
    np.save(emb_path / 'cell_embs_mean.npy', result['mean_cell_emb'])
    
    if 'context_sizes' in result:
        np.save(emb_path / 'context_sizes.npy', result['context_sizes'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings using InMemoryTokenizedDataset")
    parser.add_argument("--run_id", type=str, help="Wandb-id of the run", required=True)
    parser.add_argument("--checkpoint", type=str, default='min_val_loss.ckpt', help="checkpoint name for loading")
    parser.add_argument("--max_tokens", type=int, default=None, help="number of tokens to use")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size to use")
    parser.add_argument("--dataset", type=str, default=None, required=True, help="path to the dataset")
    parser.add_argument("--data_dir", type=str, default='h5ads', help="adata directory name")
    parser.add_argument("--filename", type=str, default='adata_0.h5ad', help="adata filename to use")
    parser.add_argument("--gene_sampling_strategy", type=str, default=None, help="gene sampling strategy to use")
    parser.add_argument("--entity", type=str, default='theislab-transformer', help="wandb entity to use")
    parser.add_argument("--project", type=str, default='contrastive-transformer', help="wandb project to use")
    args = parser.parse_args()

    # Print GPU info if available
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"GPU Type: {gpu_info.name}")
    else:
        print("Running on CPU")
    
    # Create scConcept instance to get config and determine parameters
    concept = scConcept(entity=args.entity, project=args.project)
    concept.load_config_and_model(run_id=args.run_id, checkpoint=args.checkpoint)
        
    print(f"Getting {args.data_dir}/{args.filename} embeddings for run {args.run_id} with max_tokens={args.max_tokens}, batch_size={args.batch_size}, gene_sampling_strategy={args.gene_sampling_strategy}")
    
    # Generate embeddings
    get_embs(
        run_id=args.run_id,
        dataset=args.dataset,
        filename=args.filename,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        gene_sampling_strategy=args.gene_sampling_strategy,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        entity=args.entity,
        project=args.project
    )
    
    print("Done!")

# Usage example:
# python src/concept/get_embs_inmemory.py \
#     --run_id YOUR_RUN_ID \
#     --checkpoint min_val_loss.ckpt \
#     --dataset YOUR_DATASET \
#     --filename adata_0.h5ad \
#     --batch_size 32 \
#
# Or use the class directly:
# from concept.scConcept import scConcept
# concept = scConcept()
# concept.load_config_and_model(run_id='YOUR_RUN_ID')
# embeddings = concept.extract_embeddings(adata, batch_size=32)
#
# Or with custom config and checkpoint:
# concept = scConcept(cfg=my_config, cache_dir='./cache/')
# # If using custom checkpoint path, you'll need to manually construct the path
# # and use load_config_and_model directly
# # concept.load_config_and_model(run_id='YOUR_RUN_ID', checkpoint='checkpoint.ckpt', ckpt_path='path/to/checkpoint.ckpt')
# embeddings = concept.extract_embeddings(adata, batch_size=32)