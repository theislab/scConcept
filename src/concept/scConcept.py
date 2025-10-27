import os
import shutil
import torch
from omegaconf import OmegaConf, DictConfig
from lamin_dataloader.dataset import GeneIdTokenizer
from lamin_dataloader.dataset import TokenizedDataset, BaseCollate
from lamin_dataloader.collections import InMemoryCollection
from concept.model import BiEncoderContrastiveModel
import wandb
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
                 entity: str = 'theislab-transformer', 
                 project: str = 'contrastive-transformer',
                 cache_dir: str = './cache/'):
        """
        Initialize the scConcept instance.
        
        Args:
            cfg: Configuration dictionary (if provided, will use this instead of wandb)
            entity: Wandb entity
            project: Wandb project
            cache_dir: Directory for caching checkpoints
        """
        self.entity = entity
        self.project = project
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.device = None
        self.run = None
        
    def _download_artifacts_if_needed(self, api, ckpt_dir: Path, checkpoint: str):
        """
        Download model checkpoint and gene mapping artifacts from wandb if they don't exist.
        
        Args:
            api: Wandb API instance
            ckpt_dir: Directory to cache artifacts
            checkpoint: Checkpoint filename
        """
        
        model_dir = ckpt_dir / checkpoint
        model_path = model_dir / 'model.ckpt'
        gene_mapping_path = model_dir / 'pc_gene_token_mapping.pkl'
        
        # Download checkpoint if needed
        if not model_path.exists():
            print(f"Downloading checkpoint artifact '{checkpoint}:latest' from wandb...")
            artifact = api.artifact(f'{self.entity}/{self.project}/{checkpoint}:latest')
            artifact_dir = Path(artifact.download(root=str(model_dir)))
            
            # Find and rename the downloaded checkpoint file
            downloaded_files = list(artifact_dir.glob('*.ckpt')) + list(artifact_dir.glob('*.ckpt.*'))
            if downloaded_files:
                # Assuming the first ckpt file is the checkpoint we want
                downloaded_file = downloaded_files[0]
                model_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(downloaded_file), str(model_path))
                print(f"Checkpoint renamed to {model_path}")
            else:
                raise FileNotFoundError(f"Could not find checkpoint file in downloaded artifact: {downloaded_files}")
        else:
            print(f"Checkpoint already exists at {model_path}")
        
        # Download gene mapping if needed
        if not gene_mapping_path.exists():
            print(f"Downloading gene mapping artifact 'pc_gene_token_mapping.pkl:latest' from wandb...")
            artifact = api.artifact(f'{self.entity}/{self.project}/pc_gene_token_mapping.pkl:latest')
            artifact_dir = Path(artifact.download(root=str(model_dir)))
            
            # Find and rename the downloaded gene mapping file
            downloaded_files = list(artifact_dir.glob('*.pkl'))
            if downloaded_files:
                # Assuming the first pkl file is the gene mapping we want
                downloaded_file = downloaded_files[0]
                model_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(downloaded_file), str(gene_mapping_path))
                print(f"Gene mapping renamed to {gene_mapping_path}")
            else:
                raise FileNotFoundError(f"Could not find checkpoint file in downloaded artifact: {downloaded_files}")
        else:
            print(f"Gene mapping already exists at {gene_mapping_path}")
    
    
    def load_config_and_model(self, 
                              run_id, 
                              checkpoint: str):
        """
        Load configuration from wandb and initialize the model.
        
        Args:
            run_id: Wandb run ID
            checkpoint: Checkpoint name
        """
        
        ckpt_dir = Path(self.cache_dir) / run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = ckpt_dir / checkpoint / 'model.ckpt'
        gene_mapping_path = ckpt_dir / checkpoint / 'pc_gene_token_mapping.pkl'
        
        print(f"Loading config from wandb run {run_id}...")
        wandb.login()
        api = wandb.Api()
        self.run = api.run(f'{self.entity}/{self.project}/{run_id}')
        self.cfg = DictConfig(self.run.config)
        self.cfg = self.apply_compatibility_changes(self.cfg)
        
        # Download artifacts if they don't exist
        self._download_artifacts_if_needed(api, ckpt_dir, checkpoint)
        
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
   
    def extract_embeddings(self, adata: ad.AnnData, batch_size: int, max_tokens: int = None, 
                          gene_sampling_strategy: str = None):
        """
        Extract embeddings from AnnData using the loaded model.
        
        Args:
            adata: AnnData object containing single-cell data
            batch_size: Batch size for dataloader
            max_tokens: Maximum number of tokens per cell (if None, uses config default)
            gene_sampling_strategy: Gene sampling strategy ('top-nonzero', etc.) (if None, uses config default)
            
        Returns:
            dict: Dictionary containing 'cls_cell_emb', 'mean_cell_emb', and optionally 'context_sizes'
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Determine parameters with defaults from config
        max_tokens = max_tokens if max_tokens is not None else self.cfg.datamodule.dataset.train.max_tokens
        gene_sampling_strategy = gene_sampling_strategy if gene_sampling_strategy is not None else self.cfg.datamodule.gene_sampling_strategy

        print(f"Extracting embeddings from AnnData with shape {adata.shape}")
        print(f"Parameters: max_tokens={max_tokens}, batch_size={batch_size}, gene_sampling_strategy={gene_sampling_strategy}")
        
        # Create In memory TokenizedDataset
        collection = InMemoryCollection([adata])
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
    