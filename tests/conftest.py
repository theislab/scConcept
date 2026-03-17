import anndata as ad
import numpy as np
import pytest
import torch
import pandas as pd
import os
from pathlib import Path
from omegaconf import OmegaConf
import logging
from hydra import compose, initialize
from lamin_dataloader import GeneIdTokenizer
from concept.dataset import MultiSpeciesTokenizer

logger = logging.getLogger(__name__)


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
        "ENSG00000118454",
        "ENSG00000134760",
        "ENSG00000170917",
        "ENSG00000167186",
        "ENSG00000108389",
        "ENSG00000145740",
        "ENSG00000143412",
        "ENSG00000226784",
        "ENSG00000196235",
        "ENSG00000174106",
        "ENSG00000119421",
        "ENSG00000066248",
        "ENSG00000282815",
        "ENSG00000197245",
        "ENSG00000137491",
        "ENSG00000172769",
        "ENSG00000205403",
        "ENSG00000137709",
        "ENSG00000138095",
        "ENSG00000174233",
    ]

    n_cells = 30
    np.random.seed(42)

    X = np.random.negative_binomial(n=3, p=0.7, size=(n_cells, len(real_gene_names))).astype(np.int32)

    adata = ad.AnnData(X=X)
    adata.var_names = real_gene_names
    adata.var["gene_symbols"] = real_gene_names
    adata.obs["tissue"] = np.random.choice(["blood", "brain"], size=n_cells)
    adata.obs["cell_type"] = np.random.choice(["B cell", "T cell"], size=n_cells)
    adata.uns["_organism"] = "hsapiens"
    adata.uns["_tissue"] = np.array(["blood", "brain"])

    return adata


@pytest.fixture
def tokenizer(adata):
    """Create a tokenizer for the small mock adata"""
    gene_names = adata.var_names.tolist()
    n_genes = len(gene_names)

    gene_mapping = {"<cls>": 0, "<pad>": 1, **{gene_name: i + 2 for i, gene_name in enumerate(gene_names)}}

    return GeneIdTokenizer(gene_mapping)


@pytest.fixture
def train_config(adata, tokenizer, device, tmp_path):
    """Create a training configuration and set up necessary files for train.py integration tests"""
    # Create temporary directory structure
    adata_dir = tmp_path / "h5ads"
    adata_dir.mkdir()
    panels_dir = tmp_path / "panels"
    organism_panels_dir = panels_dir / "hsapiens"
    organism_panels_dir.mkdir(parents=True)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    # Save adata as h5ad file
    adata_file = adata_dir / "test_data.h5ad"
    adata.write(adata_file)

    # Create gene mapping pickle file
    gene_mapping = tokenizer.gene_mapping
    gene_mapping_series = pd.Series(gene_mapping, name="token")
    gene_mapping_path = tmp_path / "gene_mapping.csv"
    gene_mapping_series.to_csv(gene_mapping_path, index_label="gene_id")

    pretrained_vocabulary_dir = tmp_path / "embeddings"
    pretrained_vocabulary_dir.mkdir()
    pretrained_vocabulary_path = pretrained_vocabulary_dir / "pretrained_vocabulary.csv"
    gene_names = list(tokenizer.gene_mapping.keys())[2:]  # skip <pad> and <cls>
    vectors = [np.random.rand(10) for _ in gene_names]
    df = pd.DataFrame(vectors, index=gene_names)
    df.to_csv(pretrained_vocabulary_path)

    # Create a simple panel file
    panel_file = organism_panels_dir / "test_panel.csv"
    panel_df = pd.DataFrame({"Ensembl_ID": adata.var_names[:5].tolist()})
    panel_df.to_csv(panel_file, index=False)

    config_path = "../src/concept/conf"

    # Load base config using Hydra with overrides for testing
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(
            config_name="config",
            overrides=[
                # Override paths
                f"PATH.PROJECT_PATH={tmp_path}",
                f"PATH.CHECKPOINT_ROOT={checkpoint_dir}",
                f"PATH.PANELS_PATH={panels_dir}",
                f"PATH.PRETRAINED_VOCABULARY={pretrained_vocabulary_dir}",
                # Override datamodule settings
                "datamodule.obs_keys=[]",
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
                f"model.training.freeze_pretrained_vocabulary={bool(pretrained_vocabulary_path)}",
                "model.training.use_learnable_embs_freq=0.8",
                # Disable wandb
                "wandb.enabled=False",
                "wandb.run_name=test_name",
            ],
        )

    # Override GENE_MAPPING_PATHS to use only the single test species, replacing
    # the default multi-species interpolated paths from config.yaml.
    OmegaConf.update(cfg, "PATH.GENE_MAPPING_PATHS", {"hsapiens": str(gene_mapping_path)}, merge=False)

    # Override the split with a split-config dict so that resolve_split_list can
    # extract the species and build the metadata dict correctly.
    OmegaConf.update(
        cfg,
        "datamodule.dataset.train.split",
        [{"source_path": str(adata_dir), "species": "hsapiens", "train": ["test_data.h5ad"], "val": []}],
        merge=False,
    )
    logger.info(OmegaConf.to_yaml(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)))

    return cfg
