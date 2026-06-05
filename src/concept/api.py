import logging
import multiprocessing
import os
import shutil
from pathlib import Path
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
from .decoder import MLPDecoderModel, TransformerDecoderModel
from .model import ContrastiveModel
from .utils import (
    build_species_gene_mappings,
    infer_species,
    load_gene_name_to_id_mapping,
    create_placeholder_vocabulary,
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
        self.decoder_model = None
        self.tokenizer = None
        self.device = None
        self.gene_mappings_path = None

    def _download_files_if_needed(self, model_name: str, model_dir: Path, decoder_model_name: str = "mlp-nb.ckpt"):
        """
        Download model checkpoint, optional decoder checkpoint, config, per-species gene mapping CSVs,
        panels, and pretrained vocabulary from HuggingFace Hub if they don't exist.

        Args:
            model_name: Model name (e.g., 'corpus40M-model30M')
            model_dir: Directory to cache downloaded files
            decoder_model_name: Name of the decoder model checkpoint to download (default: 'mlp-nb.ckpt')
        """
        model_path = model_dir / "model.ckpt"
        gene_mappings_path = model_dir / "gene_mappings"
        config_path = model_dir / "config.yaml"
        panels_dir = model_dir / "panels"
        pretrained_vocabulary_dir = model_dir / "pretrained_vocabulary"
        decoder_model_path = model_dir / "decoder" / decoder_model_name

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

        decoder_model_path_resolved = None
        try:
            repo_files = api.list_repo_files(repo_id=self.repo_id, repo_type="model")
            decoder_repo_path = f"{model_name}/decoder/{decoder_model_name}"
            if decoder_repo_path in repo_files:
                decoder_model_path.parent.mkdir(parents=True, exist_ok=True)
                if not decoder_model_path.exists():
                    logger.info(
                        "Downloading decoder model from HuggingFace Hub "
                        f"({self.repo_id}/{decoder_repo_path})..."
                    )
                    downloaded_path = hf_hub_download(
                        repo_id=self.repo_id,
                        filename=decoder_repo_path,
                        cache_dir=str(model_dir.parent),
                    )
                    if downloaded_path != str(decoder_model_path):
                        shutil.copy2(downloaded_path, str(decoder_model_path))
                    logger.info(f"Decoder model saved to {decoder_model_path}")
                else:
                    logger.info(f"Decoder model already exists at {decoder_model_path}")
                decoder_model_path_resolved = decoder_model_path
            else:
                logger.info(f"No decoder model found in HuggingFace Hub ({self.repo_id}/{decoder_repo_path})")
        except Exception as e:
            logger.warning(f"Could not download decoder model: {e}")

        return (
            model_path,
            gene_mappings_path,
            config_path,
            panels_dir,
            pretrained_vocabulary_dir_resolved,
            decoder_model_path_resolved,
        )

    def load_config_and_model(
        self,
        model_name: str = None,
        config: str | Path | dict | DictConfig = None,
        model_path: str | Path = None,
        decoder_model_name: str = "mlp-nb.ckpt",
        decoder_model_path: str | Path = None,
        gene_mappings_path: str | Path = None,
        panels_dir: str | Path = None,
        pretrained_vocabulary_path: str | Path = None,
    ):
        """
        Load configuration and initialize the model.

        Args:
            model_name: Model name to download from HuggingFace (e.g., 'corpus40M-model30M'). List of models: https://huggingface.co/theislab/scConcept/tree/main - required if directpaths are not provided
            config: Configuration - can be a path to config file (.yaml) as str, Path, a dictionary, or DictConfig.
                When used with ``model_name``, it is merged on top of the downloaded config.
            model_path: Path to model checkpoint file (.ckpt) - if provided, bypasses HuggingFace download
            decoder_model_name: Name of the decoder model checkpoint to download (default: 'mlp-nb.ckpt')
            decoder_model_path: Optional path to a decoder checkpoint file (.ckpt). If provided, the decoder is
                loaded and can be used through ``decode()``.
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
            (
                model_path,
                gene_mappings_path,
                config_path,
                panels_dir,
                pretrained_vocabulary_path,
                downloaded_decoder_model_path,
            ) = (
                self._download_files_if_needed(model_name, model_dir, decoder_model_name)
            )
            if decoder_model_path is None:
                decoder_model_path = downloaded_decoder_model_path
            base_config = self.load_config(config_path)
            config = {} if config is None else config
            self.cfg = OmegaConf.merge(base_config, OmegaConf.create(config))
        else:
            if not all([config, model_path, gene_mappings_path]):
                raise ValueError("If using direct paths config, model_path, and gene_mappings_path must be provided")
            self.cfg = self.load_config(config)

        self.model_name = model_name
        self.panels_dir = panels_dir

        # Get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load gene mapping and build tokenizer
        gene_mappings_path = Path(str(gene_mappings_path))
        self.gene_mappings_path = gene_mappings_path
        # Multi-species: load one {species}.csv per species listed in cfg.datamodule.species
        self.tokenizer = MultiSpeciesTokenizer(build_species_gene_mappings(str(gene_mappings_path), self.cfg.datamodule.species))

        pretrained_vocabularies = None
        if pretrained_vocabulary_path is not None:
            pretrained_vocabularies = create_placeholder_vocabulary(self.tokenizer, self.cfg.model.dim_pretrained_vocab)

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

        self.model = self.model.to(self.device)
        self.model.eval()

        if decoder_model_path is not None:
            self.decoder_model = self._load_decoder_model(decoder_model_path)
            self.decoder_model = self.decoder_model.to(self.device)
            self.decoder_model.eval()

        logger.info(f"Model loaded successfully on {self.device}")

    def _load_decoder_model(self, decoder_model_path: str | Path):
        checkpoint = torch.load(decoder_model_path, map_location="cpu")
        model_type = checkpoint.get("model_type")

        if model_type == "transformer":
            decoder_cls = TransformerDecoderModel
        elif model_type == "mlp":
            decoder_cls = MLPDecoderModel
        else:
            raise ValueError(
                f"Decoder checkpoint must contain model_type as 'transformer' or 'mlp': {decoder_model_path}"
            )

        return decoder_cls.load_from_checkpoint(str(decoder_model_path), map_location="cpu")

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

    def _check_decoder_loaded(self):
        if self.decoder_model is None:
            raise ValueError("Decoder model not loaded. Call load_config_and_model(decoder_model_path=...) first.")

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

    def map_gene_ids_to_names(self, species: str, gene_ids: list[str]) -> list[object]:
        """Map gene IDs to gene names case-insensitively, using ``nan`` for unavailable gene IDs."""
        import numpy as np

        self._check_model_loaded()
        gene_name_to_id = self.get_gene_name_to_id_mapping(species)
        casefolded_gene_id_to_name: dict[str, str] = {}
        for gene_name, gene_id in gene_name_to_id.items():
            casefolded_gene_id_to_name.setdefault(gene_id.casefold(), gene_name)

        return [casefolded_gene_id_to_name.get(gene_id.casefold(), np.nan) for gene_id in gene_ids]

    def decode(
        self,
        cell_embeddings: torch.Tensor | np.ndarray,
        batch_size: int = 128,
    ) -> dict[str, torch.Tensor | list[str]]:
        """Decode cell embeddings into gene expression predictions."""
        self._check_decoder_loaded()
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if isinstance(cell_embeddings, np.ndarray):
            cell_embeddings = torch.from_numpy(cell_embeddings)
        elif not isinstance(cell_embeddings, torch.Tensor):
            cell_embeddings = torch.as_tensor(cell_embeddings)

        if cell_embeddings.ndim == 1:
            cell_embeddings = cell_embeddings.unsqueeze(0)
        elif cell_embeddings.ndim != 2:
            raise ValueError("cell_embeddings must be a 1- or 2-dimensional tensor or array")

        self.decoder_model = self.decoder_model.to(self.device)
        self.decoder_model.eval()

        cell_embeddings = cell_embeddings.to(self.device).float()
        base_gene_indices = torch.arange(self.decoder_model.num_genes, device=self.device)
        predictions = []
        gene_indices = []
        theta_params = []

        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                for start in range(0, cell_embeddings.shape[0], batch_size):
                    batch_embeddings = cell_embeddings[start : start + batch_size]
                    batch_gene_indices = base_gene_indices.unsqueeze(0).expand(batch_embeddings.shape[0], -1)
                    batch_output = self.decoder_model.decode(batch_embeddings, batch_gene_indices)
                    predictions.append(batch_output["predictions"].detach().cpu().float())
                    gene_indices.append(batch_output["gene_indices"].detach().cpu())
                    if "theta" in batch_output:
                        theta_params.append(batch_output["theta"].detach().cpu().float())

        result = {
            "predictions": torch.cat(predictions, dim=0),
            "gene_indices": torch.cat(gene_indices, dim=0),
            "output_gene_ids": self.decoder_model.output_gene_ids,
        }
        if theta_params:
            result["theta"] = torch.cat(theta_params, dim=0)

        return result

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
        use_learnable_embs: bool = True,
        return_type: str = "numpy",
        num_workers: int = 8,
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
            use_learnable_embs: Whether to enable specie-specific learnable embeddings during prediction for maximum single-species performance.
                Set to 'False' for better cross-species alignment. (default: 'True')
            return_type: Output type for embeddings: ``"numpy"`` (default) or ``"torch"``.

        Returns:
            dict: Dictionary containing 'cls_cell_emb', and optionally 'context_sizes'
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_config_and_model() first.")
        if return_type not in {"numpy", "torch"}:
            raise ValueError("return_type must be either 'numpy' or 'torch'")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Determine parameters with defaults from config
        max_tokens = max_tokens if max_tokens is not None else self.cfg.datamodule.dataset.train.max_tokens
        gene_sampling_strategy = (
            gene_sampling_strategy if gene_sampling_strategy is not None else self.cfg.datamodule.gene_sampling_strategy
        )

        logger.info(f"Extracting embeddings from AnnData with shape {adata.shape}")
        logger.info(
            "Parameters: max_tokens=%s, batch_size=%s, gene_sampling_strategy=%s, use_learnable_embs=%s",
            max_tokens,
            batch_size,
            gene_sampling_strategy,
            use_learnable_embs,
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
            show_coverage="verbose",
        )

        # Create BaseCollate
        collate_fn = BaseCollate(
            self.tokenizer.PAD_TOKEN, max_tokens=max_tokens, gene_sampling_strategy=gene_sampling_strategy
        )
        num_workers = min(int(os.getenv("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count())), num_workers)

        # Create DataLoader
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, drop_last=False, num_workers=num_workers
        )

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
                    output = self.model.predict_step(batch, batch_idx, use_learnable_embs=use_learnable_embs)

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

        return result

    def estimate_mutual_information(
        self,
        cell_embs_1: torch.Tensor,
        cell_embs_2: torch.Tensor,
    ) -> float:
        """
        Estimate mutual information between two aligned sets of cell embeddings.

        The estimate follows the contrastive objective used by scConcept:
        ``log(n_cells) - cross_entropy(normalized_embs_1 @ normalized_embs_2.T * logit_scale, arange(n_cells))``.

        Args:
            cell_embs_1: Tensor of shape ``(n_cells, embedding_dim)`` for the first panel/view.
            cell_embs_2: Tensor of shape ``(n_cells, embedding_dim)`` for the second panel/view.

        Returns:
            Estimated mutual information as ``log(n_cells) - contrastive_loss``.
        """
        self._check_model_loaded()
        if cell_embs_1.ndim != 2 or cell_embs_2.ndim != 2:
            raise ValueError("cell_embs_1 and cell_embs_2 must both be 2-dimensional tensors")
        if cell_embs_1.shape != cell_embs_2.shape:
            raise ValueError("cell_embs_1 and cell_embs_2 must have the same shape")
        if cell_embs_1.shape[0] < 1:
            raise ValueError("cell_embs_1 and cell_embs_2 must contain at least one cell")

        cell_embs_1 = cell_embs_1.to(self.device)
        cell_embs_2 = cell_embs_2.to(self.device)
        n_cells = cell_embs_1.shape[0]
        logit_scale = self.model.logit_scale.exp().to(self.device)

        with torch.no_grad():
            logits = torch.mm(cell_embs_1, cell_embs_2.t()) * logit_scale
            targets = torch.arange(n_cells, device=self.device)
            loss = F.cross_entropy(logits, targets)

        return float(np.log(n_cells) - loss.cpu().item())

    def train(self, adata_list, species=None, max_steps=None, batch_size=None, num_workers: int = 8):
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
            num_workers: Number of data loading workers for training. Clamped to available SLURM CPUs
                by the datamodule, matching ``extract_embeddings`` behavior. Defaults to 8.

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
        self.cfg.datamodule.dataloader.train.num_workers = num_workers

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
        dataset_kwargs["val"] = None
        dataloader_kwargs["val"] = None

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
