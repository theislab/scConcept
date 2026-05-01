import logging
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Sequence

import anndata as ad
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from lamin_dataloader import BaseCollate, InMemoryCollection
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import AnnDataModule
from .dataset import MultiSpeciesTokenizedDataset, MultiSpeciesTokenizer
from .model import ContrastiveModel
from .utils import (
    build_species_gene_mappings,
    infer_species,
    load_gene_name_to_id_mapping,
    load_pretrained_vocabulary,
)

logger = logging.getLogger(__name__)


class scConcept:
    """
    High-level interface for loading, adapting, and applying scConcept models.

    This wrapper handles model/config loading from Hugging Face or local paths,
    gene-tokenizer setup across species, embedding extraction from ``AnnData``,
    and optional lightweight fine-tuning on user-provided datasets.
    """

    def __init__(self, cfg: DictConfig = None, repo_id: str = "theislab/scConcept", cache_dir: str = "./cache/"):
        """
        Initialize the scConcept instance.

        Args:
            cfg: Configuration dictionary (if provided, will use this instead of HuggingFace)
            repo_id: HuggingFace repository ID
            cache_dir: Directory for caching downloaded files
        """
        self.repo_id = repo_id
        self.cfg = self.load_config(cfg) if cfg is not None else None
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.device = None
        self.gene_mappings_path = None

    def _download_files_if_needed(self, model_name: str, model_dir: Path):
        """
        Download model checkpoint, config, per-species gene mapping CSVs, panels, and pretrained vocabulary from HuggingFace Hub if they don't exist.

        Args:
            model_name: Model name (e.g., 'corpus40M-model30M')
            model_dir: Directory to cache downloaded files
        """
        model_path = model_dir / "model.ckpt"
        gene_mappings_path = model_dir / "gene_mappings"
        config_path = model_dir / "config.yaml"
        panels_dir = model_dir / "panels"
        pretrained_vocabulary_dir = model_dir / "pretrained_vocabulary"

        model_dir.mkdir(parents=True, exist_ok=True)

        # Download checkpoint if needed
        if not model_path.exists():
            logger.info(f"Downloading model.ckpt from HuggingFace Hub ({self.repo_id}/{model_name}/model.ckpt)...")
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=f"{model_name}/model.ckpt",
                cache_dir=str(model_dir.parent),
            )
            # Move downloaded file to expected location
            if downloaded_path != str(model_path):
                shutil.copy2(downloaded_path, str(model_path))
            logger.info(f"Checkpoint saved to {model_path}")
        else:
            logger.info(f"Checkpoint already exists at {model_path}")

        # Download gene mappings directory if needed
        api = HfApi()
        gene_mappings_path.mkdir(parents=True, exist_ok=True)
        try:
            repo_files = api.list_repo_files(repo_id=self.repo_id, repo_type="model")
            gm_files = [f for f in repo_files if f.startswith(f"{model_name}/gene_mappings/") and f.endswith(".csv")]
            if gm_files:
                logger.info(
                    f"Downloading gene_mappings directory from HuggingFace Hub ({self.repo_id}/{model_name}/gene_mappings/)..."
                )
                for gm_file in gm_files:
                    gm_filename = os.path.basename(gm_file)
                    gm_path = gene_mappings_path / gm_filename
                    if not gm_path.exists():
                        downloaded_path = hf_hub_download(
                            repo_id=self.repo_id,
                            filename=gm_file,
                            cache_dir=str(model_dir.parent),
                        )
                        if downloaded_path != str(gm_path):
                            shutil.copy2(downloaded_path, str(gm_path))
                logger.info(f"Gene mappings saved to {gene_mappings_path}")
            else:
                logger.warning(f"No gene mapping CSV files found in HuggingFace Hub ({self.repo_id}/{model_name}/gene_mappings/)")
        except Exception as e:
            logger.warning(f"Could not download gene mappings directory: {e}")

        # Download config if needed
        if not config_path.exists():
            logger.info(f"Downloading config.yaml from HuggingFace Hub ({self.repo_id}/{model_name}/config.yaml)...")
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=f"{model_name}/config.yaml",
                cache_dir=str(model_dir.parent),
            )
            # Move downloaded file to expected location
            if downloaded_path != str(config_path):
                shutil.copy2(downloaded_path, str(config_path))
            logger.info(f"Config saved to {config_path}")
        else:
            logger.info(f"Config already exists at {config_path}")

        # Download panels directory if needed
        panels_dir.mkdir(parents=True, exist_ok=True)
        try:
            # List all files in the panels directory on HuggingFace
            repo_files = api.list_repo_files(repo_id=self.repo_id, repo_type="model")
            panel_files = [f for f in repo_files if f.startswith(f"{model_name}/panels/") and f.endswith(".csv")]

            if panel_files:
                logger.info(
                    f"Downloading panels directory from HuggingFace Hub ({self.repo_id}/{model_name}/panels/)..."
                )
                for panel_file in panel_files:
                    panel_filename = os.path.basename(panel_file)
                    panel_path = panels_dir / panel_filename

                    if not panel_path.exists():
                        downloaded_path = hf_hub_download(
                            repo_id=self.repo_id,
                            filename=panel_file,
                            cache_dir=str(model_dir.parent),
                        )
                        # Move downloaded file to expected location
                        if downloaded_path != str(panel_path):
                            shutil.copy2(downloaded_path, str(panel_path))
                    else:
                        pass
                logger.info(f"Panels directory saved to {panels_dir}")
            else:
                logger.info(f"No panels found in HuggingFace Hub ({self.repo_id}/{model_name}/panels/)")
        except Exception as e:
            logger.warning(f"Could not download panels directory: {e}")
            logger.warning(f"Panels directory will be created at {panels_dir} but may be empty")

        # Download pretrained vocabulary directory if needed
        pretrained_vocabulary_dir_resolved = None
        try:
            repo_files = api.list_repo_files(repo_id=self.repo_id, repo_type="model")
            pv_files = [
                f for f in repo_files
                if f.startswith(f"{model_name}/pretrained_vocabulary/") and f.endswith(".csv")
            ]
            if pv_files:
                pretrained_vocabulary_dir.mkdir(parents=True, exist_ok=True)
                logger.info(
                    f"Downloading pretrained vocabulary from HuggingFace Hub ({self.repo_id}/{model_name}/pretrained_vocabulary/)..."
                )
                for pv_file in pv_files:
                    pv_filename = os.path.basename(pv_file)
                    pv_path = pretrained_vocabulary_dir / pv_filename
                    if not pv_path.exists():
                        downloaded_path = hf_hub_download(
                            repo_id=self.repo_id,
                            filename=pv_file,
                            cache_dir=str(model_dir.parent),
                        )
                        if downloaded_path != str(pv_path):
                            shutil.copy2(downloaded_path, str(pv_path))
                logger.info(f"Pretrained vocabulary saved to {pretrained_vocabulary_dir}")
                pretrained_vocabulary_dir_resolved = pretrained_vocabulary_dir
            else:
                logger.info(f"No pretrained vocabulary found in HuggingFace Hub ({self.repo_id}/{model_name}/pretrained_vocabulary/)")
        except Exception as e:
            logger.warning(f"Could not download pretrained vocabulary directory: {e}")

        return model_path, gene_mappings_path, config_path, panels_dir, pretrained_vocabulary_dir_resolved

    def load_config_and_model(
        self,
        model_name: str = None,
        config: str | Path | dict | DictConfig = None,
        model_path: str | Path = None,
        gene_mappings_path: str | Path = None,
        panels_dir: str | Path = None,
        pretrained_vocabulary_path: str | Path = None,
    ):
        """
        Load configuration and initialize the model.

        Args:
            model_name: Model name to download from HuggingFace (e.g., 'corpus40M-model30M'). List of models: https://huggingface.co/theislab/scConcept/tree/main - required if directpaths are not provided
            config: Configuration - can be a path to config file (.yaml) as str, Path, a dictionary, or DictConfig. If provided, bypasses HuggingFace download for config
            model_path: Path to model checkpoint file (.ckpt) - if provided, bypasses HuggingFace download
            gene_mappings_path: Path to gene mappings. For multi-species models, a directory containing
                ``{species}.csv`` files (one per species). For single-species models, a ``.pkl`` or
                ``.csv`` file. If provided, bypasses HuggingFace download
            panels_dir: Path to panels directory - if provided, bypasses HuggingFace download
            pretrained_vocabulary_path: Path to pretrained vocabulary directory (containing .csv files) - if provided, overrides config PATH.PRETRAINED_VOCABULARY
        """

        if model_name:
            # Fall back to HuggingFace download
            model_dir = Path(self.cache_dir) / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Loading config and model from HuggingFace Hub ({self.repo_id}/{model_name})...")

            # Download files if they don't exist
            model_path, gene_mappings_path, config, panels_dir, pretrained_vocabulary_path = self._download_files_if_needed(
                model_name, model_dir
            )
        else:
            if not all([config, model_path, gene_mappings_path]):
                raise ValueError("If using direct paths config, model_path and gene_mappings_path must be provided")

        self.model_name = model_name
        self.panels_dir = panels_dir

        # Load config from file or use provided dict
        if self.cfg is None:
            self.cfg = self.load_config(config)

        # Load gene mapping and build tokenizer
        gene_mappings_path = Path(str(gene_mappings_path))
        self.gene_mappings_path = gene_mappings_path
        # Multi-species: load one {species}.csv per species listed in cfg.datamodule.species
        self.tokenizer = MultiSpeciesTokenizer(build_species_gene_mappings(str(gene_mappings_path), self.cfg.datamodule.species))

        pretrained_vocabularies = None
        if pretrained_vocabulary_path is not None:
            pretrained_vocabularies = load_pretrained_vocabulary(pretrained_vocabulary_path, self.tokenizer, self.cfg.model.dim_pretrained_vocab)

        # Load model
        model_args = {
            "config": self.cfg.model,
            "pad_token_id": self.tokenizer.PAD_TOKEN,
            "cls_token_id": self.tokenizer.CLS_TOKEN,
            "vocab_sizes": self.tokenizer.vocab_sizes,
            "pretrained_vocabularies": pretrained_vocabularies,
        }
        try:
            self.model = ContrastiveModel.load_from_checkpoint(str(model_path), **model_args, strict=True)
        except Exception:
            logger.info("load_from_checkpoint failed; retrying with Fabric loader...")
            from lightning.fabric import Fabric

            fabric = Fabric(accelerator="cuda" if torch.cuda.is_available() else "cpu", devices=1)
            self.model = ContrastiveModel(**model_args)
            fabric.load(str(model_path), {"model": self.model})

        # Get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully on {self.device}")

    @staticmethod
    def load_config(config: str | Path | dict | DictConfig):
        """Load configuration from file or dict."""

        if isinstance(config, (str, Path)):
            cfg = OmegaConf.load(config)
        elif isinstance(config, dict):
            cfg = OmegaConf.create(config)
        elif isinstance(config, DictConfig):
            cfg = config
        else:
            raise ValueError("Config must be a string path, dict, or DictConfig")
        cfg = scConcept.apply_compatibility_changes(cfg)
        return cfg

    def _check_model_loaded(self):
        if self.model is None or self.cfg is None or self.gene_mappings_path is None:
            raise ValueError("Model not loaded. Call load_config_and_model() first.")

    @property
    def species(self) -> list[str]:
        """Return species names supported by the loaded model."""
        self._check_model_loaded()
        return list(self.cfg.datamodule.species)

    def get_gene_name_to_id_mapping(self, species: str) -> dict[str, str]:
        """Return a ``{gene_name: gene_id}`` mapping for one species."""
        self._check_model_loaded()
        return load_gene_name_to_id_mapping(str(self.gene_mappings_path), species)

    def map_gene_names_to_ids(self, species: str, gene_names: list[str]) -> list[object]:
        """Map gene names to gene IDs case-insensitively, using ``nan`` for unavailable gene names."""
        import numpy as np

        self._check_model_loaded()
        gene_name_to_id = self.get_gene_name_to_id_mapping(species)
        casefolded_gene_name_to_id: dict[str, str] = {}
        for gene_name, gene_id in gene_name_to_id.items():
            casefolded_gene_name_to_id.setdefault(gene_name.casefold(), gene_id)

        return [casefolded_gene_name_to_id.get(gene_name.casefold(), np.nan) for gene_name in gene_names]

    @staticmethod
    def validate_config(cfg: DictConfig):
        """Validate configuration constraints."""

        if "train" in cfg.datamodule.dataset and cfg.model.pe_max_len < cfg.datamodule.dataset.train.max_tokens + 1:
            raise ValueError(
                f"Configuration validation failed: model.pe_max_len ({cfg.model.pe_max_len}) must be greater than "
                f"datamodule.dataset.train.max_tokens ({cfg.datamodule.dataset.train.max_tokens})"
            )

    @staticmethod
    def apply_compatibility_changes(cfg: DictConfig):
        """Apply compatibility changes for older checkpoints. Returns updated cfg."""
        if "projection_dim" not in cfg.model:
            cfg.model.projection_dim = None
        if "weight_decay" not in cfg.model.training:
            cfg.model.training.weight_decay = 0.0
        if "min_lr" not in cfg.model.training:
            cfg.model.training.min_lr = 0.0
        if "data_loading_speed_sanity_check" not in cfg.model:
            cfg.model.data_loading_speed_sanity_check = False
        if "decoder_head" not in cfg.model:
            cfg.model.decoder_head = True
        if "mask_value" not in cfg.model:
            cfg.model.mask_value = -1
        if "cls_value" not in cfg.model:
            cfg.model.cls_value = -2
        if "gene_sampling_strategy" in cfg.datamodule.dataset.train:
            cfg.datamodule.gene_sampling_strategy = cfg.datamodule.dataset.train.gene_sampling_strategy
        if "gene_sampling_strategy" not in cfg.datamodule:
            cfg.datamodule.gene_sampling_strategy = "top-nonzero"
        if "model_speed_sanity_check" not in cfg.datamodule:
            cfg.datamodule.model_speed_sanity_check = False
        if "min_tokens" not in cfg.model:
            cfg.model.min_tokens = None
        if "max_tokens" not in cfg.model:
            cfg.model.max_tokens = None
        if "mask_padding" not in cfg.model:
            cfg.model.mask_padding = False
        if "flash_attention" not in cfg.model:
            cfg.model.flash_attention = False
        if "pe_max_len" not in cfg.model:
            cfg.model.pe_max_len = 5000
        if "loss_switch_step" not in cfg.model:
            cfg.model.loss_switch_step = 2000
        if "PATH" in cfg and "GENE_MAPPINGS_PATH" not in cfg.PATH and "gene_mapping_path" in cfg.PATH:
            cfg.PATH.GENE_MAPPINGS_PATH = cfg.PATH.gene_mapping_path
        if "freeze_pretrained_vocabulary" not in cfg.model.training:
            cfg.model.training.freeze_pretrained_vocabulary = None
        if "use_learnable_embs_freq" not in cfg.model.training:
            cfg.model.training.use_learnable_embs_freq = None
        return cfg

    def extract_embeddings(
        self,
        adata: ad.AnnData,
        species: str = None,
        gene_id_column: str = None,
        batch_size: int = 32,
        max_tokens: int = None,
        gene_sampling_strategy: str = None,
        return_type: str = "numpy",
    ):
        """
        Extract embeddings from AnnData using the loaded model.

        Args:
            adata: AnnData object containing single-cell data
            species: Species identifier (e.g. 'hsapiens'). If not provided, will be inferred from gene IDs using the tokenizer's gene mappings if possible.
            gene_id_column: Column name in adata.var to use as gene IDs: ENSGXXXXXXXXXXX (default: None, uses index)
            batch_size: Batch size for dataloader (default: 32)
            max_tokens: Maximum number of tokens per cell (if None, uses config default)
            gene_sampling_strategy: Gene sampling strategy ('top-nonzero', etc.) (if None, uses config default)
            return_type: Output type for embeddings: ``"numpy"`` (default) or ``"torch"``.

        Returns:
            dict: Dictionary containing 'cls_cell_emb', and optionally 'context_sizes'
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_config_and_model() first.")
        if return_type not in {"numpy", "torch"}:
            raise ValueError("return_type must be either 'numpy' or 'torch'")

        self.model.eval()

        # Determine parameters with defaults from config
        max_tokens = max_tokens if max_tokens is not None else self.cfg.datamodule.dataset.train.max_tokens
        gene_sampling_strategy = (
            gene_sampling_strategy if gene_sampling_strategy is not None else self.cfg.datamodule.gene_sampling_strategy
        )

        logger.info(f"Extracting embeddings from AnnData with shape {adata.shape}")
        logger.info(
            f"Parameters: max_tokens={max_tokens}, batch_size={batch_size}, gene_sampling_strategy={gene_sampling_strategy}"
        )

        # Resolve species
        if species is None:
            adata_genes = set(adata.var[gene_id_column] if gene_id_column is not None else adata.var_names)
            species = infer_species(adata_genes, self.tokenizer)
            logger.info(f"Inferred species '{species}' from gene overlap.")

        # Create In memory MultiSpeciesTokenizedDataset
        collection = InMemoryCollection([adata], var_column=gene_id_column)
        dataset = MultiSpeciesTokenizedDataset(
            metadata={"species": [species]},
            collection=collection,
            tokenizer=self.tokenizer,
            normalization=self.cfg.datamodule.normalization,
        )

        # Create BaseCollate
        collate_fn = BaseCollate(
            self.tokenizer.PAD_TOKEN, max_tokens=max_tokens, gene_sampling_strategy=gene_sampling_strategy
        )
        num_workers = min(int(os.getenv("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count())), 8)

        # Create DataLoader
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, drop_last=False, num_workers=num_workers
        )

        logger.info(f"Processing {len(dataset)} cells...")

        # Collect embeddings
        all_cls_embs = []
        all_context_sizes = []

        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating embeddings")):
                    # Move batch to device
                    batch = {
                        key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                        for key, value in batch.items()
                    }

                    # Call predict_step
                    output = self.model.predict_step(batch, batch_idx)

                    # Collect embeddings
                    all_cls_embs.append(output["cls_cell_emb"].detach())

                    # Store context sizes (actual number of tokens per cell)
                    if "context_sizes" in output:
                        all_context_sizes.extend(output["context_sizes"])

        # Concatenate all embeddings
        cls_cell_embs = torch.cat(all_cls_embs, dim=0)
        if return_type == "numpy":
            cls_cell_embs = cls_cell_embs.cpu().detach().numpy()

        result = {"cls_cell_emb": cls_cell_embs}

        if all_context_sizes:
            result["context_sizes"] = all_context_sizes

        logger.info(f"Extracted embeddings with shape: cls={cls_cell_embs.shape}")
        logger.info(f"Total cells processed: {len(cls_cell_embs)}")

        return result

    @staticmethod
    def _normalize_panel_features(panel: Sequence[str] | np.ndarray | str) -> np.ndarray | None:
        """Return panel features as string IDs; ``"all_genes"`` is resolved after adata feature IDs are known."""
        if isinstance(panel, str):
            if panel != "all_genes":
                raise ValueError("Panel strings are only supported for 'all_genes'. Pass gene IDs as a sequence.")
            return None

        features = np.asarray(panel, dtype=str)
        if features.ndim != 1:
            raise ValueError("panel must be a one-dimensional sequence of gene IDs")
        return features

    @staticmethod
    def _subset_adata_to_panel(
        adata: ad.AnnData,
        panel: Sequence[str] | np.ndarray | str,
        gene_id_column: str = None,
    ) -> tuple[ad.AnnData, np.ndarray, int]:
        """Subset ``adata`` to genes present in ``panel`` and return matched features plus original panel size."""
        if gene_id_column is None:
            adata_features = adata.var_names.to_numpy(dtype=str)
        else:
            if gene_id_column not in adata.var:
                raise ValueError(f"gene_id_column '{gene_id_column}' is not present in adata.var")
            adata_features = adata.var[gene_id_column].to_numpy(dtype=str)

        panel_features = scConcept._normalize_panel_features(panel)
        if panel_features is None:
            panel_features = adata_features

        panel_size = int(len(panel_features))
        if panel_size == 0:
            raise ValueError("panel must contain at least one feature")

        panel_feature_set = set(panel_features)
        feature_mask = np.asarray([feature in panel_feature_set for feature in adata_features], dtype=bool)
        matched_features = adata_features[feature_mask]
        if len(matched_features) == 0:
            raise ValueError("panel has no features in common with adata")

        return adata[:, feature_mask], matched_features, panel_size

    def estimate_mutual_information(
        self,
        adata: ad.AnnData,
        panel_1: Sequence[str] | np.ndarray | str,
        panel_2: Sequence[str] | np.ndarray | str,
        species: str = None,
        gene_id_column: str = None,
        batch_size: int = 32,
        estimate_size: int = None,
        random_seed: int = 42,
    ) -> dict[str, object]:
        """
        Estimate mutual information between two gene panels over the same cells.

        The estimate follows the contrastive objective used by scConcept:
        ``log(n_cells) - cross_entropy(normalized_embs_1 @ normalized_embs_2.T * logit_scale, arange(n_cells))``.

        Args:
            adata: AnnData object containing the cells to compare.
            panel_1: First panel as gene IDs matching ``adata.var_names`` (or ``gene_id_column``), or ``"all_genes"``.
            panel_2: Second panel as gene IDs matching ``adata.var_names`` (or ``gene_id_column``), or ``"all_genes"``.
            species: Species identifier. If None, inferred separately by ``extract_embeddings``.
            gene_id_column: Column in ``adata.var`` used as gene IDs. If None, uses ``adata.var_names``.
            batch_size: Batch size for embedding extraction.
            estimate_size: Optional number of cells randomly sampled before embedding extraction.
            random_seed: Seed for estimate-size cell subsampling.

        Returns:
            dict: Dictionary with the following entries:

                - ``mutual_info``: Estimated mutual information, computed as
                  ``log(n_cells) - contrastive_loss``.
                - ``contrastive_loss``: Cross-entropy loss over the panel-pair logits, using each cell's
                  matching cell in the other panel view as the positive target.
                - ``n_cells``: Number of cells used for the estimate. This equals ``estimate_size`` when
                  provided, otherwise ``adata.n_obs``.
                - ``panel_size_1``: Number of input features requested in ``panel_1`` before intersecting
                  with ``adata``.
                - ``panel_size_2``: Number of input features requested in ``panel_2`` before intersecting
                  with ``adata``.
                - ``features_size_1``: Number of ``panel_1`` features found in ``adata`` and used for
                  embedding extraction.
                - ``features_size_2``: Number of ``panel_2`` features found in ``adata`` and used for
                  embedding extraction.
                - ``features_1``: Feature IDs from ``panel_1`` that were present in ``adata``, ordered as
                  in ``adata.var``.
                - ``features_2``: Feature IDs from ``panel_2`` that were present in ``adata``, ordered as
                  in ``adata.var``.
        """
        self._check_model_loaded()
        if adata.n_obs < 1:
            raise ValueError("adata must contain at least one cell")

        if estimate_size is not None:
            if estimate_size < 1:
                raise ValueError("estimate_size must be a positive integer")
            if estimate_size > adata.n_obs:
                raise ValueError(f"estimate_size ({estimate_size}) cannot exceed adata.n_obs ({adata.n_obs})")
            rng = np.random.default_rng(seed=random_seed)
            adata = adata[rng.choice(adata.n_obs, estimate_size, replace=False)].copy()

        adata_1, features_1, panel_size_1 = self._subset_adata_to_panel(adata, panel_1, gene_id_column)
        adata_2, features_2, panel_size_2 = self._subset_adata_to_panel(adata, panel_2, gene_id_column)

        embedding_kwargs = {
            "species": species,
            "gene_id_column": gene_id_column,
            "batch_size": batch_size,
            "return_type": "torch",
        }
        cell_embs_1 = self.extract_embeddings(adata_1, **embedding_kwargs)["cls_cell_emb"]
        cell_embs_2 = self.extract_embeddings(adata_2, **embedding_kwargs)["cls_cell_emb"]

        if len(cell_embs_1) != len(cell_embs_2):
            raise ValueError("Panel embeddings must contain the same number of cells")

        n_cells = len(cell_embs_1)
        logit_scale = self.model.logit_scale.exp().to(self.device)

        with torch.no_grad():
            logits = torch.mm(cell_embs_1, cell_embs_2.t()) * logit_scale
            targets = torch.arange(n_cells, device=self.device)
            loss = F.cross_entropy(logits, targets)

        contrastive_loss = float(loss.cpu().item())
        mutual_info = float(np.log(n_cells) - contrastive_loss)

        return {
            "mutual_info": mutual_info,
            "contrastive_loss": contrastive_loss,
            "n_cells": n_cells,
            "panel_size_1": panel_size_1,
            "panel_size_2": panel_size_2,
            "features_size_1": int(len(features_1)),
            "features_size_2": int(len(features_2)),
            "features_1": features_1,
            "features_2": features_2,
        }

    def train(self, adata_list, species=None, max_steps=None, batch_size=None):
        """Train a new model using the configuration in self.cfg.

        Uses self.model if it exists, otherwise initializes a new model.
        Assumes single GPU device with num_nodes=1.

        Args:
            adata_list: A single AnnData object or file path string, or a list of these.
            species: Species identifier(s). A single string when adata_list is a single item, or
                a list of strings with the same length as adata_list when it is a list. If None
                (or a list containing None entries), species will be inferred from gene ID overlap
                with the tokenizer vocabularies — inference is only possible for AnnData items,
                not file path strings.
            max_steps: Optional maximum number of training steps. If provided, overrides config value.
            batch_size: Optional batch size for training. If provided, overrides config value.

        Examples::

            # Single AnnData — species inferred automatically
            model.train(adata)

            # Single AnnData — species provided explicitly
            model.train(adata, species="hsapiens")

            # List of AnnData — all species inferred
            model.train([adata1, adata2])

            # List of AnnData — all species provided explicitly
            model.train([adata1, adata2], species=["hsapiens", "mmusculus"])

            # List of AnnData — mixed: first inferred, second explicit
            model.train([adata1, adata2], species=[None, "mmusculus"])

            # File path strings — species must always be provided explicitly
            model.train("path/to/data.h5ad", species="hsapiens")
            model.train(["path/to/a.h5ad", "path/to/b.h5ad"], species=["hsapiens", "mmusculus"])
        """
        if self.cfg is None:
            raise ValueError("Configuration not loaded. Set self.cfg or call load_config_and_model() first.")


        logger.info("Starting training...")

        adaptaion = self.model is not None

        panels_dir = self.panels_dir if adaptaion else self.cfg.PATH.PANELS_PATH

        # Override config values if provided
        if max_steps is not None:
            self.cfg.model.training.max_steps = max_steps
        if batch_size is not None:
            if "train" not in self.cfg.datamodule.dataloader:
                self.cfg.datamodule.dataloader.train = {}
            self.cfg.datamodule.dataloader.train.batch_size = batch_size

        dataset_kwargs = OmegaConf.to_container(self.cfg.datamodule.dataset, resolve=True, throw_on_missing=True)
        dataloader_kwargs = OmegaConf.to_container(self.cfg.datamodule.dataloader, resolve=True, throw_on_missing=True)

        # Build tokenizer before species inference (inference requires vocabulary lookups)
        if adaptaion:
            assert self.tokenizer is not None, "Tokenizer not found. Please load the model first."
        else:
            self.gene_mappings_path = Path(str(self.cfg.PATH.GENE_MAPPINGS_PATH))
            self.tokenizer = MultiSpeciesTokenizer(
                build_species_gene_mappings(self.cfg.PATH.GENE_MAPPINGS_PATH, self.cfg.datamodule.species)
            )

        # Normalize adata_list and species to parallel lists
        if isinstance(adata_list, (ad.AnnData, str)):
            adata_list = [adata_list]
            species = [species]
        elif isinstance(adata_list, list):
            if len(adata_list) == 0:
                raise ValueError("adata_list cannot be empty")
            if not all(isinstance(a, (ad.AnnData, str)) for a in adata_list):
                raise ValueError("All items in adata_list must be AnnData objects or file path strings")
            if species is None:
                species = [None] * len(adata_list)
            elif isinstance(species, str):
                raise ValueError("When adata_list is a list, species must also be a list.")
            elif len(species) != len(adata_list):
                raise ValueError(
                    f"species list length ({len(species)}) must match adata_list length ({len(adata_list)})"
                )
        else:
            raise ValueError("adata_list must be an AnnData object, a file path string, or a list of these")

        # Infer species where not provided (only possible for AnnData items, not file paths)
        species_list = []
        for item, sp in zip(adata_list, species):
            if sp is None:
                if isinstance(item, str):
                    raise ValueError(
                        "Cannot infer species for file path entries. Please provide species explicitly."
                    )
                adata_genes = set(item.var_names)
                sp = infer_species(adata_genes, self.tokenizer)
                logger.info(f"Inferred species '{sp}' from gene overlap.")
            species_list.append(sp)

        if adaptaion:
            assert set(species_list).issubset(set(self.cfg.datamodule.species)), (
                f"species {species_list} must be a subset of cfg.datamodule.species {self.cfg.datamodule.species}"
            )
        dataset_kwargs["train"]["split"] = {
            "paths": adata_list,
            "metadata": {"species": species_list},
        }

        # Create datamodule
        datamodule_args = {
            "panels_path": panels_dir,
            "tokenizer": self.tokenizer,
            "obs_keys": [],
            "precomp_embs_key": self.cfg.datamodule.precomp_embs_key,
            "normalization": self.cfg.datamodule.normalization,
            "gene_sampling_strategy": self.cfg.datamodule.gene_sampling_strategy,
            "dataset_kwargs": dataset_kwargs,
            "dataloader_kwargs": dataloader_kwargs,
            "val_loader_names": [],
        }
        datamodule = AnnDataModule(**datamodule_args)

        trainer = L.Trainer(
            max_steps=self.cfg.model.training.max_steps,
            logger=False,
            accelerator="auto",
            devices="auto",
            num_nodes=1,
            limit_train_batches=float(self.cfg.model.training.limit_train_batches),
            precision="bf16-mixed",
            accumulate_grad_batches=self.cfg.model.training.accumulate_grad_batches,
        )

        # Use existing model or create new one
        if adaptaion:
            # Update model's world_size if needed
            if hasattr(self.model, "world_size"):
                self.model.world_size = 1
            if hasattr(self.model, "val_loader_names"):
                self.model.val_loader_names = []
            self.model.train()
        else:
            model_args = {
                "config": self.cfg.model,
                "pad_token_id": self.tokenizer.PAD_TOKEN,
                "cls_token_id": self.tokenizer.CLS_TOKEN,
                "vocab_sizes": self.tokenizer.vocab_sizes,
                "world_size": 1,
                "val_loader_names": [],
            }
            self.model = ContrastiveModel(**model_args)

        # Train
        trainer.fit(model=self.model, datamodule=datamodule)

        logger.info("Training completed!")
