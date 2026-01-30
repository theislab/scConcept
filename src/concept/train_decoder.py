"""Training script for decoder models (Transformer or MLP)."""

import argparse
from pathlib import Path

import anndata as ad
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from concept.decoder_model import MLPDecoderModel, TransformerDecoderModel


class GeneExpressionDataset(Dataset):
    """Dataset for gene expression prediction."""

    def __init__(self, adata: ad.AnnData, cell_emb_key: str, gene_id_key: str, layer_key: str | None = None):
        """
        Initialize dataset from AnnData.

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object with expression counts in .X and cell embeddings in .obsm
        cell_emb_key : str
            Key for cell embeddings in adata.obsm
        gene_id_key : str
            Key for gene IDs in adata.var (use "index" to use adata.var.index)
        layer_key : str | None
            Optional key for expression data in adata.layers. If None, uses adata.X
        """
        self.cell_embeddings = torch.from_numpy(adata.obsm[cell_emb_key]).float()

        # Handle layer_key: use adata.layers[layer_key] if provided, otherwise adata.X
        if layer_key is not None:
            expression_data = adata.layers[layer_key]
        else:
            expression_data = adata.X

        # Convert to dense array if sparse
        if hasattr(expression_data, 'todense'):
            self.expressions = torch.from_numpy(np.array(expression_data.todense())).float()
        else:
            self.expressions = torch.from_numpy(np.array(expression_data)).float()

        # Handle gene_id_key being "index" or a column name
        if gene_id_key == "index":
            self.gene_ids = adata.var.index.values
        else:
            self.gene_ids = adata.var[gene_id_key].values

        self.num_genes = len(self.gene_ids)

        # Create gene ID to index mapping
        self.gene_to_idx = {gene_id: idx for idx, gene_id in enumerate(self.gene_ids)}
        self.gene_indices = torch.arange(self.num_genes)

    def __len__(self):
        return len(self.cell_embeddings)

    def __getitem__(self, idx):
        return {
            "cell_embedding": self.cell_embeddings[idx],
            "gene_indices": self.gene_indices,
            "expressions": self.expressions[idx],
        }


def train_decoder(
    adata: ad.AnnData,
    cell_emb_key: str,
    gene_id_key: str,
    layer_key: str | None = None,
    output_dir: str = "./decoder_checkpoints",
    model_type: str = "transformer",
    dim_model: int = 128,
    num_head: int = 8,
    dim_hid: int = 256,
    nlayers: int = 6,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.1,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    batch_size: int = 32,
    max_epochs: int = 10,
    val_split: float = 0.1,
    num_workers: int = 4,
):
    """
    Train decoder model on AnnData.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with expression counts and cell embeddings
    cell_emb_key : str
        Key for cell embeddings in adata.obsm
    gene_id_key : str
        Key for gene IDs in adata.var (use "index" to use adata.var.index)
    layer_key : str | None
        Optional key for expression data in adata.layers. If None, uses adata.X
    output_dir : str
        Directory to save checkpoints
    model_type : str
        Type of decoder model: "transformer" or "mlp"
    dim_model : int
        Transformer model dimension (only used for transformer model)
    num_head : int
        Number of attention heads (only used for transformer model)
    dim_hid : int
        Hidden dimension for feedforward network (only used for transformer model)
    nlayers : int
        Number of transformer decoder layers (only used for transformer model)
    hidden_dims : list[int] | None
        Hidden layer dimensions for MLP model (only used for mlp model)
        If None, defaults to [512, 512, 512]
    dropout : float
        Dropout rate
    lr : float
        Learning rate
    weight_decay : float
        Weight decay for optimizer
    batch_size : int
        Batch size for training
    max_epochs : int
        Maximum number of training epochs
    val_split : float
        Fraction of data to use for validation
    num_workers : int
        Number of workers for data loading
    """
    print(f"Using AnnData with {adata.n_obs} cells x {adata.n_vars} genes")

    # Create dataset
    dataset = GeneExpressionDataset(adata, cell_emb_key, gene_id_key, layer_key)

    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Initialize model
    cell_emb_dim = dataset.cell_embeddings.shape[1]
    num_genes = dataset.num_genes

    print(f"Initializing {model_type} model with cell_emb_dim={cell_emb_dim}, num_genes={num_genes}")

    if model_type == "transformer":
        model = TransformerDecoderModel(
            num_genes=num_genes,
            cell_emb_dim=cell_emb_dim,
            dim_model=dim_model,
            num_head=num_head,
            dim_hid=dim_hid,
            nlayers=nlayers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
        )
    elif model_type == "mlp":
        model = MLPDecoderModel(
            num_genes=num_genes,
            cell_emb_dim=cell_emb_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'transformer' or 'mlp'")

    # Setup trainer
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename="decoder-{epoch:02d}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
    )

    # Use CUDA if available, otherwise CPU
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices="auto",
        callbacks=[checkpoint_callback],
        default_root_dir=output_dir,
        log_every_n_steps=10,
    )

    # Train
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    print(f"Training complete! Best model saved to {checkpoint_callback.best_model_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train gene expression decoder model")

    # Data arguments
    parser.add_argument("--adata_path", type=str, required=True, help="Path to AnnData file")
    parser.add_argument("--cell_emb_key", type=str, required=True, help="Key for cell embeddings in adata.obsm")
    parser.add_argument("--gene_id_key", type=str, default="gene_ids", help="Key for gene IDs in adata.var (use 'index' for adata.var.index)")
    parser.add_argument("--layer_key", type=str, default=None, help="Optional key for expression data in adata.layers (if not provided, uses adata.X)")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./decoder_checkpoints", help="Output directory for checkpoints")

    # Model arguments
    parser.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "mlp"], help="Type of decoder model")
    parser.add_argument("--dim_model", type=int, default=128, help="Transformer model dimension (transformer only)")
    parser.add_argument("--num_head", type=int, default=8, help="Number of attention heads (transformer only)")
    parser.add_argument("--dim_hid", type=int, default=256, help="Hidden dimension (transformer only)")
    parser.add_argument("--nlayers", type=int, default=6, help="Number of decoder layers (transformer only)")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=None, help="Hidden layer dimensions for MLP (mlp only), e.g., --hidden_dims 512 512 512")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training arguments
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")

    args = parser.parse_args()

    # Load AnnData
    print(f"Loading AnnData from {args.adata_path}...")
    adata = ad.read_h5ad(args.adata_path)

    train_decoder(
        adata=adata,
        cell_emb_key=args.cell_emb_key,
        gene_id_key=args.gene_id_key,
        layer_key=args.layer_key,
        output_dir=args.output_dir,
        model_type=args.model_type,
        dim_model=args.dim_model,
        num_head=args.num_head,
        dim_hid=args.dim_hid,
        nlayers=args.nlayers,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
