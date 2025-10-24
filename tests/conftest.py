import anndata as ad
import numpy as np
import pytest
import torch
from lamin_dataloader.dataset import GeneIdTokenizer


def get_device():
    """Get the best available device (CUDA if available, otherwise CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@pytest.fixture
def device():
    """Fixture providing the best available device for tests"""
    return get_device()


@pytest.fixture
def adata():
    """Create a smaller mock AnnData object for testing with different batch sizes"""
    n_cells = 10
    n_genes = 30
    np.random.seed(42)
    
    X = np.random.negative_binomial(n=3, p=0.7, size=(n_cells, n_genes)).astype(np.int32)
    gene_names = [f'GENE_{i:03d}' for i in range(n_genes)]
    
    adata = ad.AnnData(X=X)
    adata.var_names = gene_names
    adata.var['gene_symbols'] = gene_names
    
    return adata


@pytest.fixture
def tokenizer(adata):
    """Create a tokenizer for the small mock adata"""
    gene_names = adata.var_names.tolist()
    n_genes = len(gene_names)
    
    gene_mapping = {
        '<pad>': 0,
        '<cls>': 1,
        **{f'GENE_{i:03d}': i + 2 for i in range(n_genes)}
    }
    
    return GeneIdTokenizer(gene_mapping)
