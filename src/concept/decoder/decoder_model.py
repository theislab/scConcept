"""
Transformer Decoder model for gene expression prediction.

Input: sequence where position 0 is cell embedding, positions 1+ are gene embeddings
Output: scalar gene expression count for each gene
"""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from concept.modules.flash_attention_layer import FlashTransformerDecoderLayer
from concept.modules.transformer import TransformerDecoder


def _normalize_reconstruction_loss(reconstruction_loss: str) -> str:
    if reconstruction_loss == "nb":
        return "negative_binomial"
    if reconstruction_loss not in {"mse", "negative_binomial"}:
        raise ValueError(
            f"Unknown reconstruction_loss: {reconstruction_loss}. Must be 'mse', 'nb', or 'negative_binomial'"
        )
    return reconstruction_loss


def _normalize_output_gene_ids(output_gene_ids: list[str]) -> list[str]:
    output_gene_ids = [str(gene_id) for gene_id in output_gene_ids]
    if len(output_gene_ids) == 0:
        raise ValueError("output_gene_ids must contain at least one gene")
    return output_gene_ids


def _log_nb_positive(x: Tensor, mu: Tensor, theta: Tensor, eps: float = 1e-8) -> Tensor:
    """Negative binomial log-likelihood with mean/dispersion parameterization."""
    log_theta_mu_eps = torch.log(theta + mu + eps)
    return (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )


def _reconstruction_loss(
    predictions: Tensor,
    target_expressions: Tensor,
    loss_fn: str,
    theta: Tensor | None = None,
) -> Tensor:
    target_expressions = target_expressions.float()

    if loss_fn == "mse":
        predictions = F.softplus(predictions.float())
        return F.mse_loss(predictions, target_expressions)
    if loss_fn == "negative_binomial":
        if theta is None:
            raise ValueError("theta must be provided for negative_binomial loss")
        library_size = target_expressions.sum(dim=-1, keepdim=True)
        mu = library_size.float() * F.softmax(predictions.float(), dim=-1)
        theta = F.softplus(theta.float())
        return -_log_nb_positive(target_expressions, mu, theta).mean()

    raise ValueError(f"Unknown reconstruction_loss: {loss_fn}. Must be 'mse' or 'negative_binomial'")


def _total_count_normalize(values: Tensor, target_sum: float) -> Tensor:
    total_counts = values.sum(dim=-1, keepdim=True)
    normalized = values / total_counts.clamp_min(torch.finfo(values.dtype).tiny) * target_sum
    return torch.where(total_counts > 0, normalized, values)


def _decode_predictions(
    predictions: Tensor,
    reconstruction_loss: str,
    theta: Tensor | None = None,
) -> dict[str, Tensor]:
    result = {"predictions": predictions}
    if reconstruction_loss == "mse":
        predictions = torch.expm1(F.softplus(predictions.float()))
        result["predictions"] = _total_count_normalize(predictions, target_sum=1.0)
    if reconstruction_loss == "negative_binomial":
        if theta is None:
            raise ValueError("theta must be provided for negative_binomial decode")
        mu = F.softmax(predictions.float(), dim=-1)
        result["theta"] = F.softplus(theta)
        result["predictions"] = mu
    return result


class MLPDecoderModel(L.LightningModule):
    """Simple MLP decoder for gene expression prediction.

    Takes cell embedding and directly predicts expression for all genes using MLP layers.
    """

    def __init__(
        self,
        output_gene_ids: list[str],
        cell_emb_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        reconstruction_loss: str = "mse",
    ):
        super().__init__()
        output_gene_ids = _normalize_output_gene_ids(output_gene_ids)
        num_genes = len(output_gene_ids)
        model_type = "mlp"
        self.save_hyperparameters()

        self.num_genes = num_genes
        self.model_type = model_type
        self.cell_emb_dim = cell_emb_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.reconstruction_loss = _normalize_reconstruction_loss(reconstruction_loss)
        self.output_gene_ids = output_gene_ids

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
        self.nb_log_theta = (
            nn.Parameter(torch.zeros(num_genes)) if self.reconstruction_loss == "negative_binomial" else None
        )

    def _normalize_gene_indices(self, cell_embedding: Tensor, gene_indices: Tensor | None = None) -> Tensor:
        if gene_indices is None:
            gene_indices = torch.arange(self.num_genes, device=cell_embedding.device)

        if gene_indices.dim() == 1:
            gene_indices = gene_indices.unsqueeze(0).expand(cell_embedding.shape[0], -1)

        return gene_indices

    def _gather_gene_values(self, values: Tensor, gene_indices: Tensor) -> Tensor:
        if values.shape == gene_indices.shape:
            return values
        if values.shape[-1] != self.num_genes:
            raise ValueError(
                "values must either match gene_indices shape or contain one value per output gene "
                f"(got values shape {tuple(values.shape)} and gene_indices shape {tuple(gene_indices.shape)})"
            )
        return torch.gather(values, dim=-1, index=gene_indices)

    def forward(self, cell_embedding: Tensor, gene_indices: Tensor | None = None) -> Tensor:
        """
        Forward pass.

        Args:
            cell_embedding: Cell embedding of shape (batch_size, cell_emb_dim)
            gene_indices: Optional gene indices of shape (num_genes_in_batch,) or
                (batch_size, num_genes_in_batch). If provided, predictions are
                subset to these genes.

        Returns
        -------
        predictions: Gene expression predictions of shape (batch_size, num_genes_in_batch)
            when gene_indices is provided, otherwise (batch_size, num_genes)
        """
        # MLP directly predicts all gene expressions from cell embedding
        predictions = self.mlp(cell_embedding)  # (batch_size, num_genes)
        if gene_indices is not None:
            gene_indices = self._normalize_gene_indices(cell_embedding, gene_indices)
            predictions = self._gather_gene_values(predictions, gene_indices)
        return predictions

    def _loss(
        self,
        predictions: Tensor,
        target_expressions: Tensor,
        gene_indices: Tensor | None = None,
    ) -> Tensor:
        if gene_indices is not None:
            target_expressions = self._gather_gene_values(target_expressions, gene_indices)
            theta = self.nb_log_theta[gene_indices] if self.nb_log_theta is not None else None
        else:
            theta = self.nb_log_theta.unsqueeze(0).expand_as(predictions) if self.nb_log_theta is not None else None
        return _reconstruction_loss(predictions, target_expressions, self.reconstruction_loss, theta)

    def decode(
        self,
        cell_embedding: Tensor,
        gene_indices: Tensor | None = None,
    ) -> dict[str, Tensor]:
        gene_indices = self._normalize_gene_indices(cell_embedding, gene_indices)
        predictions = self(cell_embedding, gene_indices)
        theta = self.nb_log_theta[gene_indices] if self.nb_log_theta is not None else None
        result = _decode_predictions(predictions, self.reconstruction_loss, theta)
        result["gene_indices"] = gene_indices
        return result

    def training_step(self, batch, batch_idx):
        """Training step."""
        cell_embedding = batch["cell_embedding"]  # (batch_size, cell_emb_dim)
        gene_indices = batch.get("gene_indices")
        if gene_indices is not None:
            gene_indices = self._normalize_gene_indices(cell_embedding, gene_indices)
        target_expressions = batch["expressions"]  # (batch_size, num_genes)

        predictions = self(cell_embedding, gene_indices)
        loss = self._loss(predictions, target_expressions, gene_indices)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        cell_embedding = batch["cell_embedding"]  # (batch_size, cell_emb_dim)
        gene_indices = batch.get("gene_indices")
        if gene_indices is not None:
            gene_indices = self._normalize_gene_indices(cell_embedding, gene_indices)
        target_expressions = batch["expressions"]  # (batch_size, num_genes)

        predictions = self(cell_embedding, gene_indices)
        loss = self._loss(predictions, target_expressions, gene_indices)

        self.log("val/loss", loss, prog_bar=True)
        return loss

    def on_save_checkpoint(self, checkpoint):
        checkpoint["model_type"] = self.model_type

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
        output_gene_ids: list[str],
        cell_emb_dim: int,
        dim_model: int = 128,
        num_head: int = 8,
        dim_hid: int = 256,
        nlayers: int = 6,
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        use_flash_attn: bool | None = None,
        reconstruction_loss: str = "mse",
    ):
        super().__init__()
        output_gene_ids = _normalize_output_gene_ids(output_gene_ids)
        num_genes = len(output_gene_ids)
        model_type = "transformer"
        self.save_hyperparameters()

        self.num_genes = num_genes
        self.model_type = model_type
        self.cell_emb_dim = cell_emb_dim
        self.dim_model = dim_model
        self.lr = lr
        self.weight_decay = weight_decay
        self.reconstruction_loss = _normalize_reconstruction_loss(reconstruction_loss)
        self.output_gene_ids = output_gene_ids
        if use_flash_attn is None:
            use_flash_attn = torch.cuda.is_available()

        # Cell embedding adaptation layer
        self.cell_emb_adapter = nn.Linear(cell_emb_dim, dim_model)

        # Trainable gene embedding table
        self.gene_embeddings = nn.Embedding(num_genes, dim_model)

        # Transformer decoder
        decoder_layer = FlashTransformerDecoderLayer(
            d_model=dim_model,
            nhead=num_head,
            dim_feedforward=dim_hid,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
        )
        decoder_norm = nn.LayerNorm(dim_model)
        self.decoder = TransformerDecoder(decoder_layer, nlayers, decoder_norm)

        # Expression prediction head: maps from dim_model to scalar
        self.expression_head = nn.Linear(dim_model, 1)
        self.nb_log_theta = (
            nn.Parameter(torch.zeros(num_genes)) if self.reconstruction_loss == "negative_binomial" else None
        )

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

    def _loss(
        self,
        predictions: Tensor,
        target_expressions: Tensor,
        gene_indices: Tensor,
    ) -> Tensor:
        theta = self.nb_log_theta[gene_indices] if self.nb_log_theta is not None else None
        return _reconstruction_loss(predictions, target_expressions, self.reconstruction_loss, theta)

    def decode(
        self,
        cell_embedding: Tensor,
        gene_indices: Tensor,
    ) -> dict[str, Tensor]:
        predictions = self(cell_embedding, gene_indices)
        theta = self.nb_log_theta[gene_indices] if self.nb_log_theta is not None else None
        result = _decode_predictions(predictions, self.reconstruction_loss, theta)
        result["gene_indices"] = gene_indices
        return result

    def training_step(self, batch, batch_idx):
        """Training step."""
        cell_embedding = batch["cell_embedding"]  # (batch_size, cell_emb_dim)
        gene_indices = batch["gene_indices"]  # (batch_size, num_genes)
        target_expressions = batch["expressions"]  # (batch_size, num_genes)

        predictions = self(cell_embedding, gene_indices)
        loss = self._loss(predictions, target_expressions, gene_indices)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        cell_embedding = batch["cell_embedding"]  # (batch_size, cell_emb_dim)
        gene_indices = batch["gene_indices"]  # (batch_size, num_genes)
        target_expressions = batch["expressions"]  # (batch_size, num_genes)

        predictions = self(cell_embedding, gene_indices)
        loss = self._loss(predictions, target_expressions, gene_indices)

        self.log("val/loss", loss, prog_bar=True)
        return loss

    def on_save_checkpoint(self, checkpoint):
        checkpoint["model_type"] = self.model_type

    def configure_optimizers(self):
        """Configure AdamW optimizer with cosine annealing."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
