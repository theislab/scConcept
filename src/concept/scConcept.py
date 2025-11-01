import os
import shutil
import torch
from omegaconf import OmegaConf, DictConfig
from lamin_dataloader.dataset import GeneIdTokenizer
from lamin_dataloader.dataset import TokenizedDataset, BaseCollate
from lamin_dataloader.collections import InMemoryCollection
from concept.model import BiEncoderContrastiveModel
from huggingface_hub import hf_hub_download
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
        Download model checkpoint, config, and gene mapping files from HuggingFace Hub if they don't exist.
        
        Args:
            model_name: Model name (e.g., 'Corpus-30M')
            model_dir: Directory to cache downloaded files
        """
        model_path = model_dir / 'model.ckpt'
        gene_mapping_path = model_dir / 'pc_gene_token_mapping.pkl'
        config_path = model_dir / 'config.yaml'
        
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
        
        return model_path, gene_mapping_path, config_path
    
    
    def load_config_and_model(self, 
                              model_name: str):
        """
        Load configuration from HuggingFace Hub and initialize the model.
        
        Args:
            model_name: Model name (e.g., 'Corpus-30M')
        """
        
        model_dir = Path(self.cache_dir) / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading config and model from HuggingFace Hub ({self.repo_id}/{model_name})...")
        
        # Download files if they don't exist
        model_path, gene_mapping_path, config_path = self._download_files_if_needed(model_name, model_dir)
        
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
    