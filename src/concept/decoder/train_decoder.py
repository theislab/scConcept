"""Training script for decoder models (Transformer or MLP)."""

import argparse
from numbers import Integral, Real
from pathlib import Path

import anndata as ad
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from concept.decoder.decoder_model import (
    MLPDecoderModel,
    TransformerDecoderModel,
    _normalize_reconstruction_loss,
)


MSE_NORMALIZATION_TARGET_SUM = 1e4


def _normalize_mse_expressions(expressions: torch.Tensor) -> torch.Tensor:
    total_counts = expressions.sum(dim=-1, keepdim=True)
    normalized = expressions / total_counts.clamp_min(torch.finfo(expressions.dtype).tiny)
    normalized = normalized * MSE_NORMALIZATION_TARGET_SUM
    normalized = torch.where(total_counts > 0, normalized, expressions)
    return torch.log1p(normalized)


def _parse_gene_sample_size(value: str) -> int | float:
    try:
        parsed = int(value)
    except ValueError:
        parsed = float(value)
    return parsed


def _resolve_gene_sample_size(gene_sample_size: int | float | None, num_genes: int) -> int | None:
    if gene_sample_size is None:
        return None

    if isinstance(gene_sample_size, bool):
        raise TypeError("gene_sample_size must be None, a float fraction, or an integer gene count")

    if isinstance(gene_sample_size, Integral):
        sample_size = int(gene_sample_size)
        if sample_size < 1 or sample_size > num_genes:
            raise ValueError(f"integer gene_sample_size must be between 1 and {num_genes}")
        return sample_size

    if isinstance(gene_sample_size, Real):
        fraction = float(gene_sample_size)
        if fraction <= 0 or fraction > 1:
            raise ValueError("float gene_sample_size must be greater than 0 and at most 1")
        return max(1, int(num_genes * fraction))

    raise TypeError("gene_sample_size must be None, a float fraction, or an integer gene count")


class GeneExpressionDataset(Dataset):
    """Dataset for gene expression prediction."""

    def __init__(
        self,
        adata: ad.AnnData,
        cell_emb_key: str,
        gene_id_key: str,
        layer_key: str | None = None,
        gene_sample_size: int | float | None = None,
        normalize_expressions: bool = False,
    ):
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
        gene_sample_size : int | float | None
            If None, return all genes. If int, randomly sample that many genes per
            item. If float, randomly sample that fraction of genes per item.
        normalize_expressions : bool
            If True, apply total count normalization to 1e4 followed by log1p.
        """
        self.cell_embeddings = torch.from_numpy(adata.obsm[cell_emb_key]).float()
        self.normalize_expressions = normalize_expressions

        # Keep expression data in its original representation and only densify
        # the selected rows during item fetching.
        if layer_key is not None:
            self.expressions = adata.layers[layer_key]
        else:
            self.expressions = adata.X

        # Handle gene_id_key being "index" or a column name
        if gene_id_key == "index":
            self.gene_ids = adata.var.index.values
        else:
            self.gene_ids = adata.var[gene_id_key].values

        self.num_genes = len(self.gene_ids)

        # Create gene ID to index mapping
        self.gene_to_idx = {gene_id: idx for idx, gene_id in enumerate(self.gene_ids)}
        self.gene_indices = torch.arange(self.num_genes)
        self.gene_sample_size = _resolve_gene_sample_size(gene_sample_size, self.num_genes)

    def __len__(self):
        return len(self.cell_embeddings)

    def _expression_slice_to_tensor(self, idx):
        expression_slice = self.expressions[idx]

        if hasattr(expression_slice, "toarray"):
            expression_slice = expression_slice.toarray()
        elif hasattr(expression_slice, "todense"):
            expression_slice = expression_slice.todense()

        return torch.from_numpy(np.asarray(expression_slice)).float()

    def __getitem__(self, idx):
        gene_indices = self.gene_indices
        expressions = self._expression_slice_to_tensor(idx).squeeze(0)
        if self.normalize_expressions:
            expressions = _normalize_mse_expressions(expressions)
        if self.gene_sample_size is not None:
            gene_indices = torch.randperm(self.num_genes)[: self.gene_sample_size]
            expressions = expressions[gene_indices]

        return {
            "cell_embedding": self.cell_embeddings[idx],
            "gene_indices": gene_indices,
            "expressions": expressions,
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
    use_flash_attn: bool | None = None,
    reconstruction_loss: str = "mse",
    gene_sample_size: int | float | None = None,
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
    use_flash_attn : bool | None
        Whether to use FlashAttention for the transformer decoder. If None, uses
        FlashAttention only when CUDA is available.
    reconstruction_loss : str
        Reconstruction loss to use: "mse", "nb", or "negative_binomial".
    gene_sample_size : int | float | None
        If None, train on all genes. If int, randomly sample that many genes per
        dataset item. If float, randomly sample that fraction of genes per item.
    """
    print(f"Using AnnData with {adata.n_obs} cells x {adata.n_vars} genes")
    reconstruction_loss = _normalize_reconstruction_loss(reconstruction_loss)

    # Create dataset
    dataset = GeneExpressionDataset(
        adata,
        cell_emb_key,
        gene_id_key,
        layer_key,
        gene_sample_size,
        normalize_expressions=reconstruction_loss == "mse",
    )

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
    output_gene_ids = [str(gene_id) for gene_id in dataset.gene_ids]

    print(f"Initializing {model_type} model with cell_emb_dim={cell_emb_dim}, num_genes={len(output_gene_ids)}")

    if model_type == "transformer":
        model = TransformerDecoderModel(
            output_gene_ids=output_gene_ids,
            cell_emb_dim=cell_emb_dim,
            dim_model=dim_model,
            num_head=num_head,
            dim_hid=dim_hid,
            nlayers=nlayers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            use_flash_attn=use_flash_attn,
            reconstruction_loss=reconstruction_loss,
        )
    elif model_type == "mlp":
        model = MLPDecoderModel(
            output_gene_ids=output_gene_ids,
            cell_emb_dim=cell_emb_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            reconstruction_loss=reconstruction_loss,
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
        save_top_k=1,
        save_last="link",
    )

    # Use CUDA if available, otherwise CPU
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    precision = "bf16-mixed" if accelerator == "cuda" else "32-true"

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices="auto",
        callbacks=[checkpoint_callback],
        default_root_dir=output_dir,
        log_every_n_steps=10,
        precision=precision,
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
    parser.add_argument(
        "--gene_sample_size",
        type=_parse_gene_sample_size,
        default=None,
        help=(
            "Random gene subset size per dataset item. Use an integer count "
            "or a float fraction between 0 and 1. Defaults to all genes."
        ),
    )
    parser.add_argument(
        "--reconstruction_loss",
        type=str,
        default="mse",
        choices=["mse", "nb", "negative_binomial"],
        help="Reconstruction loss to use",
    )
    parser.add_argument(
        "--use_flash_attn",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Use FlashAttention for transformer decoder. Defaults to CUDA availability.",
    )

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
        use_flash_attn=args.use_flash_attn,
        reconstruction_loss=args.reconstruction_loss,
        gene_sample_size=args.gene_sample_size,
    )


if __name__ == "__main__":
    main()
