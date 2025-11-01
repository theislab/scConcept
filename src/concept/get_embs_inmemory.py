import os
import torch
from concept.scConcept import scConcept
import numpy as np
from pathlib import Path
import argparse
import anndata as ad


def get_embs(model_name: str, adata_path: str, batch_size: int,
             gene_id_column: str = None, repo_id: str = 'mojtababahrami/scConcept', 
             cache_dir: str = './cache/'):
    """
    Generate embeddings using InMemory TokenizedDataset.
    
    Args:
        model_name: Model name (e.g., 'Corpus-30M')
        adata_path: Path to the AnnData file (.h5ad)
        batch_size: Batch size for dataloader
        gene_id_column: Column name in adata.var to use as gene IDs (default: None, uses index)
        repo_id: HuggingFace repository ID
        cache_dir: Directory for caching downloaded files
    """
    
    # Create scConcept instance
    concept = scConcept(repo_id=repo_id, cache_dir=cache_dir)
    
    # Load model
    concept.load_config_and_model(model_name=model_name)
    
    # Get the directory of the adata file
    adata_path_obj = Path(adata_path)
    adata_dir = adata_path_obj.parent
    adata_filename = adata_path_obj.stem  # filename without extension
    
    # Create output directory in the same directory as the adata file
    emb_path = adata_dir / f'{adata_filename}_embs' / model_name
    
    # Check if embeddings already exist
    if (emb_path / 'cell_embs_cls.npy').exists():
        print(f"Embeddings already exist in {emb_path} ...")
        return
    
    # Load adata
    adata = ad.read_h5ad(adata_path)

    # Extract embeddings
    result = concept.extract_embeddings(
        adata=adata,
        batch_size=batch_size,
        gene_id_column=gene_id_column
    )
    
    # Save embeddings
    print(f"Saving embeddings to {emb_path}...")
    os.makedirs(emb_path, exist_ok=True)
    np.save(emb_path / 'cell_embs_cls.npy', result['cls_cell_emb'])
    np.save(emb_path / 'cell_embs_mean.npy', result['mean_cell_emb'])
    
    if 'context_sizes' in result:
        np.save(emb_path / 'context_sizes.npy', result['context_sizes'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings using InMemory TokenizedDataset")
    parser.add_argument("--model_name", type=str, help="Model name (e.g., 'Corpus-30M')", required=True)
    parser.add_argument("--adata_path", type=str, help="Path to the AnnData file (.h5ad)", required=True)
    parser.add_argument("--batch_size", type=int, default=32, help="batch size to use")
    parser.add_argument("--gene_id_column", type=str, default=None, help="Column name in adata.var to use as gene IDs (default: None, uses index)")
    parser.add_argument("--repo_id", type=str, default='mojtababahrami/scConcept', help="HuggingFace repository ID")
    parser.add_argument("--cache_dir", type=str, default='./cache/', help="Directory for caching downloaded files")
    args = parser.parse_args()

    # Print GPU info if available
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"GPU Type: {gpu_info.name}")
    else:
        print("Running on CPU")
    
    print(f"Getting embeddings for {args.adata_path} using model {args.model_name} with batch_size={args.batch_size}, gene_id_column={args.gene_id_column}")
    
    # Generate embeddings
    get_embs(
        model_name=args.model_name,
        adata_path=args.adata_path,
        batch_size=args.batch_size,
        gene_id_column=args.gene_id_column,
        repo_id=args.repo_id,
        cache_dir=args.cache_dir
    )
    
    print("Done!")

# Usage example:
# python src/concept/get_embs_inmemory.py \
#     --model_name Corpus-30M \
#     --adata_path /path/to/your/adata.h5ad \
#     --batch_size 32 \
#     --gene_id_column gene_id \
