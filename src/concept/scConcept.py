import os
import shutil
import torch
import lightning as L
from omegaconf import OmegaConf, DictConfig
from lamin_dataloader.dataset import GeneIdTokenizer
from lamin_dataloader.dataset import TokenizedDataset, BaseCollate
from lamin_dataloader.collections import InMemoryCollection
from concept.model import BiEncoderContrastiveModel
from concept.data.datamodules import AnnDataModule
from huggingface_hub import hf_hub_download, HfApi
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import anndata as ad
from torch.utils.data import DataLoader


class scConcept:
    """
    A class for loading scConcept models and extracting embeddings from single-cell data.
    """
    
    def __init__(self, 
                 cfg: DictConfig = None,
                 repo_id: str = 'theislab/scConcept',
                 cache_dir: str = './cache/'):
        """
        Initialize the scConcept instance.
        
        Args:
            cfg: Configuration dictionary (if provided, will use this instead of HuggingFace)
            repo_id: HuggingFace repository ID
            cache_dir: Directory for caching downloaded files
        """
        self.repo_id = repo_id
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def _download_files_if_needed(self, model_name: str, model_dir: Path):
        """
        Download model checkpoint, config, gene mapping files, and panels directory from HuggingFace Hub if they don't exist.
        
        Args:
            model_name: Model name (e.g., 'Corpus-30M')
            model_dir: Directory to cache downloaded files
        """
        model_path = model_dir / 'model.ckpt'
        gene_mapping_path = model_dir / 'pc_gene_token_mapping.pkl'
        config_path = model_dir / 'config.yaml'
        panels_dir = model_dir / 'panels'
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Download checkpoint if needed
        if not model_path.exists():
            print(f"Downloading model.ckpt from HuggingFace Hub ({self.repo_id}/{model_name}/model.ckpt)...")
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=f"{model_name}/model.ckpt",
                cache_dir=str(model_dir.parent),
            )
            # Move downloaded file to expected location
            if downloaded_path != str(model_path):
                shutil.copy2(downloaded_path, str(model_path))
            print(f"Checkpoint saved to {model_path}")
        else:
            print(f"Checkpoint already exists at {model_path}")
        
        # Download gene mapping if needed
        if not gene_mapping_path.exists():
            print(f"Downloading pc_gene_token_mapping.pkl from HuggingFace Hub ({self.repo_id}/{model_name}/pc_gene_token_mapping.pkl)...")
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=f"{model_name}/pc_gene_token_mapping.pkl",
                cache_dir=str(model_dir.parent),
            )
            # Move downloaded file to expected location
            if downloaded_path != str(gene_mapping_path):
                shutil.copy2(downloaded_path, str(gene_mapping_path))
            print(f"Gene mapping saved to {gene_mapping_path}")
        else:
            print(f"Gene mapping already exists at {gene_mapping_path}")
        
        # Download config if needed
        if not config_path.exists():
            print(f"Downloading config.yaml from HuggingFace Hub ({self.repo_id}/{model_name}/config.yaml)...")
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=f"{model_name}/config.yaml",
                cache_dir=str(model_dir.parent),
            )
            # Move downloaded file to expected location
            if downloaded_path != str(config_path):
                shutil.copy2(downloaded_path, str(config_path))
            print(f"Config saved to {config_path}")
        else:
            print(f"Config already exists at {config_path}")
        
        # Download panels directory if needed
        panels_dir.mkdir(parents=True, exist_ok=True)
        api = HfApi()
        try:
            # List all files in the panels directory on HuggingFace
            repo_files = api.list_repo_files(repo_id=self.repo_id, repo_type="model")
            panel_files = [f for f in repo_files if f.startswith(f"{model_name}/panels/") and f.endswith(".csv")]
            
            if panel_files:
                print(f"Downloading panels directory from HuggingFace Hub ({self.repo_id}/{model_name}/panels/)...")
                for panel_file in panel_files:
                    panel_filename = os.path.basename(panel_file)
                    panel_path = panels_dir / panel_filename
                    
                    if not panel_path.exists():
                        print(f"  Downloading {panel_filename}...")
                        downloaded_path = hf_hub_download(
                            repo_id=self.repo_id,
                            filename=panel_file,
                            cache_dir=str(model_dir.parent),
                        )
                        # Move downloaded file to expected location
                        if downloaded_path != str(panel_path):
                            shutil.copy2(downloaded_path, str(panel_path))
                        print(f"  Panel {panel_filename} saved to {panel_path}")
                    else:
                        print(f"  Panel {panel_filename} already exists at {panel_path}")
                print(f"Panels directory saved to {panels_dir}")
            else:
                print(f"No panels found in HuggingFace Hub ({self.repo_id}/{model_name}/panels/)")
        except Exception as e:
            print(f"Warning: Could not download panels directory: {e}")
            print(f"Panels directory will be created at {panels_dir} but may be empty")
        
        return model_path, gene_mapping_path, config_path, panels_dir
    
    
    def load_config_and_model(self, 
                              model_name: str):
        """
        Load configuration from HuggingFace Hub and initialize the model.
        
        Args:
            model_name: Model name (e.g., 'Corpus-30M')
        """
        self.model_name = model_name
        
        model_dir = Path(self.cache_dir) / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading config and model from HuggingFace Hub ({self.repo_id}/{model_name})...")
        
        # Download files if they don't exist
        model_path, gene_mapping_path, config_path, panels_dir = self._download_files_if_needed(model_name, model_dir)
        
        self.panels_dir = panels_dir
        
        # Load config from downloaded file
        self.cfg = OmegaConf.load(config_path)
        self.cfg = self.apply_compatibility_changes(self.cfg)
                
        # Load gene mapping
        gene_mapping = pd.read_pickle(gene_mapping_path).to_dict()
        
        # Create tokenizer
        self.tokenizer = GeneIdTokenizer(gene_mapping)
        
        # Load model
        model_args = {
            'config': self.cfg.model,
            'pad_token_id': gene_mapping['<pad>'],
            'cls_token_id': gene_mapping['<cls>'],
            'vocab_size': len(gene_mapping),
        }
        self.model = BiEncoderContrastiveModel.load_from_checkpoint(str(model_path), **model_args)
        
        # Get device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    @staticmethod
    def apply_compatibility_changes(cfg: DictConfig):
        """Apply compatibility changes for older checkpoints. Returns updated cfg."""
        if 'per_view_normalization' not in cfg.model:
            cfg.model.per_view_normalization = False
        if 'projection_dim' not in cfg.model:
            cfg.model.projection_dim = None
        if 'weight_decay' not in cfg.model.training:
            cfg.model.training.weight_decay = 0.0
        if 'min_lr' not in cfg.model.training:
            cfg.model.training.min_lr = 0.0
        if 'data_loading_speed_sanity_check' not in cfg.model:
            cfg.model.data_loading_speed_sanity_check = False
        if 'decoder_head' not in cfg.model:
            cfg.model.decoder_head = True
        if 'gene_sampling_strategy' in cfg.datamodule.dataset.train:
            cfg.datamodule.gene_sampling_strategy = cfg.datamodule.dataset.train.gene_sampling_strategy
        if 'gene_sampling_strategy' not in cfg.datamodule:
            cfg.datamodule.gene_sampling_strategy = 'top-nonzero'
        if 'model_speed_sanity_check' not in cfg.datamodule:
            cfg.datamodule.model_speed_sanity_check = False
        if 'min_tokens' not in cfg.model:
            cfg.model.min_tokens = None
        if 'max_tokens' not in cfg.model:
            cfg.model.max_tokens = None
        if 'mask_padding' not in cfg.model:
            cfg.model.mask_padding = False
        if 'flash_attention' not in cfg.model:
            cfg.model.flash_attention = False
        if 'pe_max_len' not in cfg.model:
            cfg.model.pe_max_len = 5000
        if 'loss_switch_step' not in cfg.model:
            cfg.model.loss_switch_step = 2000
        return cfg
   
    def extract_embeddings(self, adata: ad.AnnData, gene_id_column: str = None, batch_size: int = 32, max_tokens: int = None, 
                          gene_sampling_strategy: str = None):
        """
        Extract embeddings from AnnData using the loaded model.
        
        Args:
            adata: AnnData object containing single-cell data
            gene_id_column: Column name in adata.var to use as gene IDs: ENSGXXXXXXXXXXX (default: None, uses index)
            batch_size: Batch size for dataloader (default: 32)
            max_tokens: Maximum number of tokens per cell (if None, uses config default)
            gene_sampling_strategy: Gene sampling strategy ('top-nonzero', etc.) (if None, uses config default)
            
        Returns:
            dict: Dictionary containing 'cls_cell_emb', 'mean_cell_emb', and optionally 'context_sizes'
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_config_and_model() first.")
        
        self.model.eval()
        
        # Determine parameters with defaults from config
        max_tokens = max_tokens if max_tokens is not None else self.cfg.datamodule.dataset.train.max_tokens
        gene_sampling_strategy = gene_sampling_strategy if gene_sampling_strategy is not None else self.cfg.datamodule.gene_sampling_strategy

        print(f"Extracting embeddings from AnnData with shape {adata.shape}")
        print(f"Parameters: max_tokens={max_tokens}, batch_size={batch_size}, gene_sampling_strategy={gene_sampling_strategy}")
        
        # Create In memory TokenizedDataset
        collection = InMemoryCollection([adata], var_column=gene_id_column)
        dataset = TokenizedDataset(
            collection,
            self.tokenizer,
            normalization=self.cfg.datamodule.normalization,
        )
        
        # Create BaseCollate
        collate_fn = BaseCollate(
            self.tokenizer.PAD_TOKEN,
            max_tokens=max_tokens,
            gene_sampling_strategy=gene_sampling_strategy
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        
        print(f"Processing {len(dataset)} cells...")
        
        # Collect embeddings
        all_cls_embs = []
        all_mean_embs = []
        all_context_sizes = []
        
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating embeddings")):
                    # Move batch to device
                    batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value 
                            for key, value in batch.items()}
                    
                    # Call predict_step
                    output = self.model.predict_step(batch, batch_idx)
                    
                    # Collect embeddings
                    all_cls_embs.append(output['cls_cell_emb'].cpu())
                    all_mean_embs.append(output['mean_cell_emb'].cpu())
                    
                    # Store context sizes (actual number of tokens per cell)
                    if 'context_sizes' in output:
                        all_context_sizes.extend(output['context_sizes'])
        
        # Concatenate all embeddings
        cls_cell_embs = torch.cat(all_cls_embs, dim=0).cpu().detach().numpy()
        mean_cell_embs = torch.cat(all_mean_embs, dim=0).cpu().detach().numpy()
        
        result = {
            'cls_cell_emb': cls_cell_embs,
            'mean_cell_emb': mean_cell_embs
        }
        
        if all_context_sizes:
            result['context_sizes'] = all_context_sizes
            
        print(f"Extracted embeddings with shape: cls={cls_cell_embs.shape}, mean={mean_cell_embs.shape}")
        print(f"Total cells processed: {len(cls_cell_embs)}")
        
        return result
    
    def train(self, adata_list=None, max_steps=None, batch_size=None):
        """
        Train a new model using the configuration in self.cfg.
        Uses self.model if it exists, otherwise initializes a new model.
        Assumes single GPU device with num_nodes=1.
        
        Args:
            adata_list: Optional AnnData object or list of AnnData objects to use for training.
                       If provided, will be used instead of loading from file paths.
            max_steps: Optional maximum number of training steps. If provided, overrides config value.
            batch_size: Optional batch size for training. If provided, overrides config value.
        """
        if self.cfg is None:
            raise ValueError("Configuration not loaded. Set self.cfg or call load_config_and_model() first.")
        
        print("Starting training...")
        
        adaptaion = self.model is not None
        
        panels_dir = self.panels_dir if adaptaion else self.cfg.PATH.PANELS_PATH
        
        # Override config values if provided
        if max_steps is not None:
            self.cfg.model.training.max_steps = max_steps
        if batch_size is not None:
            if 'train' not in self.cfg.datamodule.dataloader:
                self.cfg.datamodule.dataloader.train = {}
            self.cfg.datamodule.dataloader.train.batch_size = batch_size
        
        
        # Create split dictionary (only train, no validation)
        if adata_list is not None:
            # Handle single AnnData object or list of AnnData objects
            if isinstance(adata_list, ad.AnnData):
                # Convert single AnnData to list
                adata_list = [adata_list]
            elif isinstance(adata_list, list):
                # Validate list
                if len(adata_list) == 0:
                    raise ValueError("adata_list cannot be empty")
                if not all(isinstance(adata, ad.AnnData) for adata in adata_list):
                    raise ValueError("All items in adata_list must be AnnData objects")
            else:
                raise ValueError("adata_list must be an AnnData object or a list of AnnData objects")
            # Use provided AnnData objects
            split = {'train': adata_list}
        else:
            # Load from file paths
            dataset_path = self.cfg.PATH.ADATA_PATH
            split = {}
            for key, filenames in self.cfg.PATH.SPLIT.items():
                if filenames is not None and key == 'train':
                    split[key] = [os.path.join(dataset_path, file) for file in filenames]
            
        if adaptaion:
            assert self.tokenizer is not None, "Tokenizer not found. Please load the model first."
        else:
            # Load gene mapping
            gene_mapping = pd.read_pickle(self.cfg.PATH.gene_mapping_path).to_dict()
            self.tokenizer = GeneIdTokenizer(gene_mapping)
        
        # Create datamodule
        datamodule_args = {
            'split': split,
            'panels_path': panels_dir,
            'tokenizer': self.tokenizer,
            'columns': [],
            'precomp_embs_key': self.cfg.datamodule.precomp_embs_key,
            'normalization': self.cfg.datamodule.normalization,
            'gene_sampling_strategy': self.cfg.datamodule.gene_sampling_strategy,
            'dataset_kwargs': OmegaConf.to_container(self.cfg.datamodule.dataset, resolve=True, throw_on_missing=True),
            'dataloader_kwargs': OmegaConf.to_container(self.cfg.datamodule.dataloader, resolve=True, throw_on_missing=True),
            'val_loader_names': [],
        }
        datamodule = AnnDataModule(**datamodule_args)
        
        # Create trainer for single GPU (no validation)
        trainer = L.Trainer(
            max_steps=self.cfg.model.training.max_steps,
            accelerator='gpu',
            devices=1,
            num_nodes=1,
            limit_train_batches=float(self.cfg.model.training.limit_train_batches),
            precision='bf16-mixed',
            accumulate_grad_batches=self.cfg.model.training.accumulate_grad_batches,
        )
        
        # Use existing model or create new one
        if adaptaion:
            # Update model's world_size if needed
            if hasattr(self.model, 'world_size'):
                self.model.world_size = 1
            if hasattr(self.model, 'val_loader_names'):
                self.model.val_loader_names = []
            self.model.train()
        else:
            model_args = {
                'config': self.cfg.model,
                'pad_token_id': gene_mapping['<pad>'],
                'cls_token_id': gene_mapping['<cls>'],
                'vocab_size': len(gene_mapping),
                'world_size': 1,  # Single GPU
                'val_loader_names': [],  # No validation
                'precomp_embs_key': self.cfg.datamodule.precomp_embs_key,
            }
            self.model = BiEncoderContrastiveModel(**model_args)
        
        # Train
        trainer.fit(model=self.model, datamodule=datamodule)
        
        print("Training completed!")
    