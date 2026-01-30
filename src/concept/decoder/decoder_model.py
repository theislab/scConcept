"""
Transformer Decoder model for gene expression prediction.

Input: sequence where position 0 is cell embedding, positions 1+ are gene embeddings
Output: scalar gene expression count for each gene
"""

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor

from concept.modules.td_layer import TransformerDecoderLayer
from concept.modules.transformer import TransformerDecoder


class MLPDecoderModel(L.LightningModule):
    """Simple MLP decoder for gene expression prediction.

    Takes cell embedding and directly predicts expression for all genes using MLP layers.
    """

    def __init__(
        self,
        num_genes: int,
        cell_emb_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_genes = num_genes
        self.cell_emb_dim = cell_emb_dim
        self.lr = lr
        self.weight_decay = weight_decay

        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        # Build MLP layers
        layers = []
        in_dim = cell_emb_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(in_dim, num_genes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, cell_embedding: Tensor, gene_indices: Tensor | None = None) -> Tensor:
        """
        Forward pass.

        Args:
            cell_embedding: Cell embedding of shape (batch_size, cell_emb_dim)
            gene_indices: Gene indices of shape (batch_size, num_genes) - not used in MLP model,
                         included for API compatibility with TransformerDecoderModel

        Returns
        -------
        predictions: Gene expression predictions of shape (batch_size, num_genes)
        """
        # MLP directly predicts all gene expressions from cell embedding
        predictions = self.mlp(cell_embedding)  # (batch_size, num_genes)
        return predictions

    def training_step(self, batch, batch_idx):
        """Training step with MSE loss."""
        cell_embedding = batch["cell_embedding"]  # (batch_size, cell_emb_dim)
        target_expressions = batch["expressions"]  # (batch_size, num_genes)

        predictions = self(cell_embedding)
        loss = nn.functional.mse_loss(predictions, target_expressions)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        cell_embedding = batch["cell_embedding"]  # (batch_size, cell_emb_dim)
        target_expressions = batch["expressions"]  # (batch_size, num_genes)

        predictions = self(cell_embedding)
        loss = nn.functional.mse_loss(predictions, target_expressions)

        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure AdamW optimizer with cosine annealing."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer



class TransformerDecoderModel(L.LightningModule):
    """Minimal transformer decoder for gene expression prediction."""

    def __init__(
        self,
        num_genes: int,
        cell_emb_dim: int,
        dim_model: int = 128,
        num_head: int = 8,
        dim_hid: int = 256,
        nlayers: int = 6,
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_genes = num_genes
        self.cell_emb_dim = cell_emb_dim
        self.dim_model = dim_model
        self.lr = lr
        self.weight_decay = weight_decay

        # Cell embedding adaptation layer
        self.cell_emb_adapter = nn.Linear(cell_emb_dim, dim_model)

        # Trainable gene embedding table
        self.gene_embeddings = nn.Embedding(num_genes, dim_model)

        # Transformer decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=dim_model,
            nhead=num_head,
            dim_feedforward=dim_hid,
            dropout=dropout,
            norm_first=True,
        )
        decoder_norm = nn.LayerNorm(dim_model)
        self.decoder = TransformerDecoder(decoder_layer, nlayers, decoder_norm)

        # Expression prediction head: maps from dim_model to scalar
        self.expression_head = nn.Linear(dim_model, 1)

    def forward(self, cell_embedding: Tensor, gene_indices: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            cell_embedding: Cell embedding of shape (batch_size, cell_emb_dim)
            gene_indices: Gene indices of shape (batch_size, num_genes)

        Returns
        -------
        predictions: Gene expression predictions of shape (batch_size, num_genes)
        """
        # Adapt cell embedding to model dimension
        cell_emb = self.cell_emb_adapter(cell_embedding)  # (batch_size, dim_model)
        cell_emb = cell_emb.unsqueeze(1)  # (batch_size, 1, dim_model)

        # Look up gene embeddings from trainable table
        gene_embs = self.gene_embeddings(gene_indices)  # (batch_size, num_genes, dim_model)

        # Decoder: attend to cell embedding while processing gene embeddings
        decoder_out = self.decoder(
            tgt=gene_embs,
            memory=cell_emb,
        )  # (batch_size, num_genes, dim_model)

        # Predict expression values
        predictions = self.expression_head(decoder_out).squeeze(-1)  # (batch_size, num_genes)

        return predictions

    def training_step(self, batch, batch_idx):
        """Training step with MSE loss."""
        cell_embedding = batch["cell_embedding"]  # (batch_size, cell_emb_dim)
        gene_indices = batch["gene_indices"]  # (batch_size, num_genes)
        target_expressions = batch["expressions"]  # (batch_size, num_genes)

        predictions = self(cell_embedding, gene_indices)
        loss = nn.functional.mse_loss(predictions, target_expressions)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        cell_embedding = batch["cell_embedding"]  # (batch_size, cell_emb_dim)
        gene_indices = batch["gene_indices"]  # (batch_size, num_genes)
        target_expressions = batch["expressions"]  # (batch_size, num_genes)

        predictions = self(cell_embedding, gene_indices)
        loss = nn.functional.mse_loss(predictions, target_expressions)

        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure AdamW optimizer with cosine annealing."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
