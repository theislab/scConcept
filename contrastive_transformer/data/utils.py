import numpy as np
import logging
import anndata as ad
import scanpy as sc
import gc
logger = logging.getLogger(__name__)



def balance_anndata(adata, n_obs, key):
    subsets = []
    groups = adata.obs[key].unique()
    for item in groups:
        subset = sc.pp.subsample(adata[adata.obs[key] == item], n_obs=n_obs, copy=True, random_state=42)
        subsets.append(subset)
    adata_balanced = ad.concat(subsets, axis=0)
    return adata_balanced


def add_count_nnz(adata_path: str, output_path: str = None):
    # Load the AnnData object efficiently
    adata = ad.read_h5ad(adata_path)
    
    # Compute nonzero counts efficiently
    count_nnz = (adata.X != 0).sum(axis=0)
    # count_sum = adata.X.sum(axis=0)
    
    count_nnz = np.array(count_nnz).flatten()
    # count_sum = np.array(count_sum).flatten()
    
    assert "_count_nnz" not in adata.var.keys(), "Count nnz already exists in the AnnData object"
    adata.var["_count_nnz"] = count_nnz
    # adata.var["_count_sum"] = count_sum
    
    # Save the modified object
    if output_path:
        adata.write_h5ad(output_path)
    else:
        adata.write_h5ad(adata_path)
    
    del adata
    gc.collect()