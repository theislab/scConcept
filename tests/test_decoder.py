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
        num_genes=num_genes,
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


if __name__ == "__main__":
    pytest.main([__file__])
