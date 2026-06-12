"""
Integration tests for the training pipeline and model.

This test suite provides comprehensive testing of the ContrastiveModel
and the training workflow.

"""

import importlib.util
import logging
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

logger = logging.getLogger(__name__)

FLASH_ATTN_AVAILABLE = importlib.util.find_spec("flash_attn") is not None
IS_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"
FLASH_ATTENTION_PARAMS = [
    False,
    pytest.param(
        True,
        marks=pytest.mark.skipif(not FLASH_ATTN_AVAILABLE, reason="flash_attn is not installed"),
    ),
]


def _mock_log(*args, **kwargs):
    """No-op replacement for LightningModule.log in direct step tests."""
    return None


def test_decoder_reconstruction_loss_uses_all_values():
    from concept.decoder.decoder_model import _reconstruction_loss

    predictions = torch.tensor([[1.0, 10.0, 3.0]])
    target_expressions = torch.tensor([[2.0, -1.0, 1.0]])

    loss = _reconstruction_loss(predictions, target_expressions, "mse")

    assert torch.isclose(
        loss,
        torch.nn.functional.mse_loss(torch.nn.functional.softplus(predictions), target_expressions),
    )


def test_decoder_mse_decode_applies_inverse_normalization():
    from concept.decoder.decoder_model import _decode_predictions

    predictions = torch.tensor([[-1.0, 0.0, 1.0]])
    expected = torch.expm1(torch.nn.functional.softplus(predictions))
    expected = expected / expected.sum(dim=-1, keepdim=True)

    decoded = _decode_predictions(predictions, "mse")

    assert torch.allclose(decoded["predictions"], expected)


def test_decoder_negative_binomial_loss(device):
    from concept.decoder import TransformerDecoderModel

    model = TransformerDecoderModel(
        output_gene_ids=[f"gene_{idx}" for idx in range(10)],
        cell_emb_dim=16,
        dim_model=8,
        num_head=2,
        dim_hid=16,
        nlayers=1,
        reconstruction_loss="nb",
        use_flash_attn=False,
    ).to(device)

    batch = {
        "cell_embedding": torch.randn(2, 16, device=device),
        "gene_indices": torch.arange(4, device=device).unsqueeze(0).expand(2, -1),
        "expressions": torch.tensor([[0.0, 1.0, -1.0, 3.0], [2.0, -1.0, 0.0, 1.0]], device=device),
    }

    with patch.object(model, "log", _mock_log):
        loss = model.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert not torch.isnan(loss)
    assert loss.item() >= 0
    assert model.nb_log_theta is not None


def test_mlp_decoder_subsets_predictions_and_loss_by_gene_indices(device):
    from concept.decoder import MLPDecoderModel

    model = MLPDecoderModel(
        output_gene_ids=[f"gene_{idx}" for idx in range(6)],
        cell_emb_dim=4,
        hidden_dims=[8],
        dropout=0.0,
    ).to(device)

    cell_embedding = torch.randn(2, 4, device=device)
    gene_indices = torch.tensor([[0, 2, 5], [4, 1, 3]], device=device)
    target_expressions = torch.randn(2, 6, device=device)

    full_predictions = model(cell_embedding)
    subset_predictions = model(cell_embedding, gene_indices)

    expected_predictions = torch.gather(full_predictions, dim=-1, index=gene_indices)
    expected_targets = torch.gather(target_expressions, dim=-1, index=gene_indices)

    assert subset_predictions.shape == gene_indices.shape
    assert torch.equal(subset_predictions, expected_predictions)
    expected_decode_predictions = torch.expm1(torch.nn.functional.softplus(expected_predictions))
    expected_decode_predictions = expected_decode_predictions / expected_decode_predictions.sum(dim=-1, keepdim=True)
    assert torch.allclose(
        model.decode(cell_embedding, gene_indices)["predictions"],
        expected_decode_predictions,
    )
    assert torch.isclose(
        model._loss(subset_predictions, target_expressions, gene_indices),
        torch.nn.functional.mse_loss(torch.nn.functional.softplus(expected_predictions), expected_targets),
    )


def test_mlp_decoder_training_step_accepts_varying_gene_indices(device):
    from concept.decoder import MLPDecoderModel

    model = MLPDecoderModel(
        output_gene_ids=[f"gene_{idx}" for idx in range(8)],
        cell_emb_dim=5,
        hidden_dims=[8],
        dropout=0.0,
        reconstruction_loss="nb",
    ).to(device)

    batch = {
        "cell_embedding": torch.randn(3, 5, device=device),
        "gene_indices": torch.tensor([[0, 2, 7], [1, 3, 4], [6, 5, 2]], device=device),
        "expressions": torch.poisson(torch.ones(3, 8, device=device)),
    }

    with patch.object(model, "log", _mock_log):
        loss = model.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert not torch.isnan(loss)
    assert model(batch["cell_embedding"], batch["gene_indices"]).shape == batch["gene_indices"].shape


def test_api_decode_uses_autocast():
    from concept import scConcept

    class FakeDecoderModel:
        num_genes = 3
        output_gene_ids = ["gene_0", "gene_1", "gene_2"]

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def decode(self, cell_embeddings, gene_indices):
            assert torch.is_autocast_enabled(cell_embeddings.device.type)
            return {
                "predictions": torch.ones_like(gene_indices, dtype=cell_embeddings.dtype),
                "gene_indices": gene_indices,
            }

    concept = scConcept()
    concept.device = torch.device("cpu")
    concept.decoder_model = FakeDecoderModel()

    result = concept.decode(torch.randn(4, 5), batch_size=2)

    assert result["predictions"].shape == (4, 3)
    assert torch.equal(result["gene_indices"], torch.arange(3).unsqueeze(0).expand(4, -1))
    assert result["output_gene_ids"] == ["gene_0", "gene_1", "gene_2"]


def test_decoder_output_gene_ids_validation():
    from concept.decoder import TransformerDecoderModel

    model = TransformerDecoderModel(
        output_gene_ids=["gene_a", "gene_b", "gene_c"],
        cell_emb_dim=8,
        dim_model=8,
        num_head=2,
        dim_hid=16,
        nlayers=1,
        use_flash_attn=False,
    )

    assert model.output_gene_ids == ["gene_a", "gene_b", "gene_c"]
    assert model.model_type == "transformer"
    assert model.hparams["output_gene_ids"] == ["gene_a", "gene_b", "gene_c"]

    with pytest.raises(ValueError, match="output_gene_ids must contain at least one gene"):
        TransformerDecoderModel(
            output_gene_ids=[],
            cell_emb_dim=8,
            dim_model=8,
            num_head=2,
            dim_hid=16,
            nlayers=1,
            use_flash_attn=False,
        )


def test_decoder_model_integration(device):
    """Integration test for TransformerDecoderModel.

    Tests that the decoder model can:
    - Initialize with basic parameters
    - Perform forward pass with cell embeddings and gene indices
    - Execute training step with MSE loss
    - Execute validation step
    - Update parameters during training
    """
    from concept.decoder import TransformerDecoderModel

    # Model configuration
    num_genes = 100
    cell_emb_dim = 512  # Input cell embedding dimension
    dim_model = 64  # Internal model dimension
    num_head = 4
    dim_hid = 128
    nlayers = 2
    dropout = 0.1
    lr = 1e-3
    batch_size = 8
    num_genes_in_batch = 20

    # Initialize model
    model = TransformerDecoderModel(
        output_gene_ids=[f"gene_{idx}" for idx in range(num_genes)],
        cell_emb_dim=cell_emb_dim,
        dim_model=dim_model,
        num_head=num_head,
        dim_hid=dim_hid,
        nlayers=nlayers,
        dropout=dropout,
        lr=lr,
    )

    # Move model to device
    model = model.to(device)
    model.train()

    assert model is not None
    assert model.num_genes == num_genes
    assert model.cell_emb_dim == cell_emb_dim
    assert model.dim_model == dim_model

    # Create mock batch
    cell_embedding = torch.randn(batch_size, cell_emb_dim).to(device)
    gene_indices = torch.randint(0, num_genes, (batch_size, num_genes_in_batch)).to(device)
    target_expressions = torch.randn(batch_size, num_genes_in_batch).abs().to(device)  # Positive expression values

    batch = {
        "cell_embedding": cell_embedding,
        "gene_indices": gene_indices,
        "expressions": target_expressions,
    }

    # Test forward pass
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            predictions = model(cell_embedding, gene_indices)
            assert predictions.shape == (batch_size, num_genes_in_batch)
            assert predictions.device.type == device.type
            assert not torch.isnan(predictions).any()

    # Test training step
    optimizer = model.configure_optimizers()

    # Store initial parameters
    params_before = {name: param.clone().detach() for name, param in model.named_parameters()}

    # Mock the log method to avoid trainer reference errors
    with patch.object(model, "log", _mock_log):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss = model.training_step(batch, batch_idx=0)

    # Verify loss
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert not torch.isnan(loss)
    assert loss.device.type == device.type
    assert loss.item() >= 0  # MSE loss should be non-negative

    # Backward and optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Verify parameters were updated
    params_after = {name: param for name, param in model.named_parameters()}
    parameters_updated = False
    for name, before in params_before.items():
        param = params_after[name]
        if param.requires_grad and before.dtype.is_floating_point:
            if not torch.equal(before, param):
                parameters_updated = True
                break

    assert parameters_updated, "Model parameters should be updated after training step"

    # Test validation step
    model.eval()
    with torch.no_grad(), patch.object(model, "log", _mock_log):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            val_loss = model.validation_step(batch, batch_idx=0)

    assert isinstance(val_loss, torch.Tensor)
    assert not torch.isnan(val_loss)
    assert val_loss.device.type == device.type
    assert val_loss.item() >= 0


def test_gene_expression_dataset_gene_sampling_none_keeps_all_genes(adata):
    from concept.decoder.train_decoder import GeneExpressionDataset

    adata.obsm["X_embedding"] = np.random.randn(adata.n_obs, 8).astype(np.float32)
    adata.var["gene_ids"] = adata.var_names

    dataset = GeneExpressionDataset(
        adata=adata,
        cell_emb_key="X_embedding",
        gene_id_key="gene_ids",
        gene_sample_size=None,
    )
    item = dataset[0]

    assert item["gene_indices"].shape == (adata.n_vars,)
    assert item["expressions"].shape == (adata.n_vars,)
    assert torch.equal(item["gene_indices"], torch.arange(adata.n_vars))
    assert torch.equal(item["expressions"], torch.from_numpy(np.asarray(adata.X[0])).float())


def test_gene_expression_dataset_normalizes_mse_expressions_before_gene_sampling(adata):
    from concept.decoder.train_decoder import GeneExpressionDataset

    adata.obsm["X_embedding"] = np.random.randn(adata.n_obs, 8).astype(np.float32)
    adata.var["gene_ids"] = adata.var_names

    dataset = GeneExpressionDataset(
        adata=adata,
        cell_emb_key="X_embedding",
        gene_id_key="gene_ids",
        gene_sample_size=5,
        normalize_expressions=True,
    )
    item = dataset[0]

    raw_expressions = torch.from_numpy(np.asarray(adata.X[0])).float()
    expected_expressions = torch.log1p(raw_expressions / raw_expressions.sum() * 1e4)

    assert item["expressions"].shape == (5,)
    assert torch.allclose(item["expressions"], expected_expressions[item["gene_indices"]])


def test_gene_expression_dataset_gene_sampling_integer_and_fraction(adata):
    from concept.decoder.train_decoder import GeneExpressionDataset

    adata.obsm["X_embedding"] = np.random.randn(adata.n_obs, 8).astype(np.float32)
    adata.var["gene_ids"] = adata.var_names

    int_dataset = GeneExpressionDataset(
        adata=adata,
        cell_emb_key="X_embedding",
        gene_id_key="gene_ids",
        gene_sample_size=5,
    )
    int_item = int_dataset[0]
    expected_int_expressions = torch.from_numpy(np.asarray(adata.X[0])).float()[int_item["gene_indices"]]

    assert int_item["gene_indices"].shape == (5,)
    assert int_item["expressions"].shape == (5,)
    assert torch.equal(int_item["expressions"], expected_int_expressions)

    fraction_dataset = GeneExpressionDataset(
        adata=adata,
        cell_emb_key="X_embedding",
        gene_id_key="gene_ids",
        gene_sample_size=0.25,
    )
    fraction_item = fraction_dataset[0]
    expected_fraction_size = int(adata.n_vars * 0.25)
    expected_fraction_expressions = torch.from_numpy(np.asarray(adata.X[0])).float()[fraction_item["gene_indices"]]

    assert fraction_item["gene_indices"].shape == (expected_fraction_size,)
    assert fraction_item["expressions"].shape == (expected_fraction_size,)
    assert torch.equal(fraction_item["expressions"], expected_fraction_expressions)


@pytest.mark.parametrize("gene_sample_size", [0, -1, 21, 0.0, -0.1, 1.1, True, "3"])
def test_gene_expression_dataset_gene_sampling_validation(adata, gene_sample_size):
    from concept.decoder.train_decoder import GeneExpressionDataset

    adata.obsm["X_embedding"] = np.random.randn(adata.n_obs, 8).astype(np.float32)
    adata.var["gene_ids"] = adata.var_names

    with pytest.raises((TypeError, ValueError)):
        GeneExpressionDataset(
            adata=adata,
            cell_emb_key="X_embedding",
            gene_id_key="gene_ids",
            gene_sample_size=gene_sample_size,
        )


def test_train_decoder_integration(adata, device, tmp_path):
    """Integration test for train_decoder function.

    Tests that train_decoder can:
    - Load AnnData with cell embeddings and expression data
    - Create dataset and dataloaders
    - Initialize decoder model
    - Train for a few epochs
    - Save checkpoints
    """
    from pathlib import Path

    from concept.decoder.train_decoder import train_decoder

    # Add cell embeddings to adata
    cell_emb_dim = 64
    adata.obsm["X_embedding"] = np.random.randn(adata.n_obs, cell_emb_dim).astype(np.float32)

    # Add gene IDs to var
    adata.var["gene_ids"] = adata.var_names

    # Set output directory
    output_dir = str(tmp_path / "decoder_checkpoints")

    # Train with minimal configuration for fast testing
    train_decoder(
        adata=adata,
        cell_emb_key="X_embedding",
        gene_id_key="gene_ids",
        output_dir=output_dir,
        dim_model=16,
        num_head=2,
        dim_hid=32,
        nlayers=2,
        dropout=0.1,
        lr=1e-3,
        weight_decay=0.0,
        batch_size=8,
        max_epochs=2,
        val_split=0.2,
        num_workers=0,  # Use 0 workers for simpler testing
    )

    # Verify checkpoints were created
    checkpoint_dir = Path(output_dir)
    assert checkpoint_dir.exists()
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    assert len(checkpoint_files) > 0, "No checkpoint files were created"

    checkpoint = torch.load(checkpoint_files[0], map_location="cpu")
    assert checkpoint["hyper_parameters"]["output_gene_ids"] == list(map(str, adata.var["gene_ids"]))
    assert checkpoint["model_type"] == "transformer"
    assert len(checkpoint["hyper_parameters"]["output_gene_ids"]) == adata.n_vars


if __name__ == "__main__":
    pytest.main([__file__])
