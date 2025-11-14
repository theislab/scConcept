import anndata as ad
import numpy as np
import pytest
import torch
import pandas as pd
import os
from pathlib import Path
from omegaconf import OmegaConf
from hydra import compose, initialize
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
    
    n_cells = 30
    np.random.seed(42)
    
    X = np.random.negative_binomial(n=3, p=0.7, size=(n_cells, len(real_gene_names))).astype(np.int32)
    
    adata = ad.AnnData(X=X)
    adata.var_names = real_gene_names
    adata.var['gene_symbols'] = real_gene_names
    
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


@pytest.fixture
def train_config(adata, tokenizer, device, tmp_path):
    """Create a training configuration and set up necessary files for train.py integration tests"""
    # Create temporary directory structure
    adata_dir = tmp_path / "h5ads"
    adata_dir.mkdir()
    panels_dir = tmp_path / "panels"
    panels_dir.mkdir()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    
    # Save adata as h5ad file
    adata_file = adata_dir / "test_data.h5ad"
    adata.write(adata_file)
    
    # Create gene mapping pickle file
    gene_mapping = tokenizer.gene_mapping
    gene_mapping_series = pd.Series(gene_mapping)
    gene_mapping_path = tmp_path / "gene_mapping.pkl"
    gene_mapping_series.to_pickle(gene_mapping_path)
    
    # Create a simple panel file
    panel_file = panels_dir / "test_panel.csv"
    panel_df = pd.DataFrame({'Ensembl_ID': adata.var_names[:5].tolist()})
    panel_df.to_csv(panel_file, index=False)
    
    
    try:
        config_path = "../src/concept/conf"
        
        # Load base config using Hydra with overrides for testing
        with initialize(version_base=None, config_path=config_path):
            cfg = compose(
                config_name="config",
                overrides=[
                    # Override paths
                    f"PATH.PROJECT_PATH={tmp_path}",
                    f"PATH.PROJECT_DATA_PATH={tmp_path}",
                    f"PATH.CHECKPOINT_ROOT={checkpoint_dir}",
                    f"PATH.DATASET_PATH={tmp_path}",
                    f"PATH.ADATA_PATH={adata_dir}",
                    f"PATH.PANELS_PATH={panels_dir}",
                    f"PATH.gene_mapping_path={gene_mapping_path}",
                    # Override split to use test data
                    "split=split_v1_test",
                    # Override datamodule settings
                    "datamodule.columns=[]",
                    "datamodule.normalization=raw",
                    "datamodule.gene_sampling_strategy=top-nonzero",
                    "datamodule.dataset.train.max_tokens=10",
                    "datamodule.dataset.train.panel_size_min=3",
                    "datamodule.dataset.val=null",
                    # Override dataloader settings
                    "datamodule.dataloader.train.batch_size=8",
                    "datamodule.dataloader.train.num_workers=2",
                    "datamodule.dataloader.val=null",
                    # Override model settings for faster testing
                    "model.dim_model=16",
                    "model.num_head=2",
                    "model.dim_hid=32",
                    "model.nlayers=2",
                    "model.loss_switch_step=1",
                    # Override training settings
                    "model.training.max_steps=5",
                    "model.training.warmup=1",
                    "model.training.devices=1",
                    "model.training.num_nodes=1",
                    # Disable wandb
                    "wandb.enabled=False",
                    "wandb.run_name=test_name",
                ]
            )
    finally:
        # Manually override the split to use our test file
        cfg.PATH.SPLIT = {'train': ['test_data.h5ad']}
        print(OmegaConf.to_yaml(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)))
    
    # Return config and checkpoint directory for verification
    return {
        'config': cfg,
        'checkpoint_dir': checkpoint_dir
    }
