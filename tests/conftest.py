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
    # Use real gene names from the provided list
    real_gene_names = [
        'ENSG00000118454', 'ENSG00000134760', 'ENSG00000170917', 
        'ENSG00000167186', 'ENSG00000108389', 'ENSG00000145740', 
        'ENSG00000143412', 'ENSG00000226784', 'ENSG00000196235', 
        'ENSG00000174106', 'ENSG00000119421', 'ENSG00000066248', 
        'ENSG00000282815', 'ENSG00000197245', 'ENSG00000137491', 
        'ENSG00000172769', 'ENSG00000205403', 'ENSG00000137709', 
        'ENSG00000138095', 'ENSG00000174233'
    ]
    
    # Repeat the real gene names to fill n_genes
    gene_names = (real_gene_names * ((n_genes // len(real_gene_names)) + 1))[:n_genes]
    
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
        **{gene_name: i + 2 for i, gene_name in enumerate(gene_names)}
    }
    
    return GeneIdTokenizer(gene_mapping)
