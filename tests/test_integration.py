"""
Integration tests for the training pipeline and model.

This test suite provides comprehensive testing of the BiEncoderContrastiveModel
and the training workflow.

Note: flash_attn and _encode are conditionally mocked:
- If flash-attn is installed: Uses real implementation
- If flash-attn is not installed: Uses mocked implementation
- Log method is always mocked to avoid trainer reference errors
"""
import pytest
import torch
import tempfile
import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Check if flash-attn is installed
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

print(f"FLASH_ATTN_AVAILABLE: {FLASH_ATTN_AVAILABLE}")

# Mock flash_attn and its submodules only if flash-attn is not installed
if not FLASH_ATTN_AVAILABLE:
    sys.modules['flash_attn'] = MagicMock()
    sys.modules['flash_attn.modules'] = MagicMock()
    sys.modules['flash_attn.modules.mha'] = MagicMock()

# Now we can import the modules
from concept.model import BiEncoderContrastiveModel
from concept.data.datamodules import MappedCollectionDataModule
from concept.data.collate import CustomCollate
from lamin_dataloader.dataset import GeneIdTokenizer, InMemoryTokenizedDataset
from lamin_dataloader.dataset import CustomCollate as LaminCustomCollate
from torch.utils.data import DataLoader


def _mock_encode(self, tokens, values, src_key_padding_mask=None):
    """
    Mock the _encode function to avoid complex transformer computations.
    Returns mock embeddings and cell embeddings with correct shapes.
    """
    batch_size = tokens.size(0)
    seq_len = tokens.size(1)
    device = tokens.device
    embs_padded = torch.randn(batch_size, seq_len, self.dim_model, device=device, dtype=torch.float32)
    cell_embs = embs_padded[:, 0, :]
    return embs_padded, cell_embs


def _mock_log(self, name, value, sync_dist=False, add_dataloader_idx=False):
    """Mock the log method to avoid trainer reference errors."""
    pass


@pytest.fixture(autouse=True)
def mock_model_methods():
    """Automatically mock model methods for all tests in this module."""
    # Always mock log to avoid trainer reference errors
    with patch.object(BiEncoderContrastiveModel, 'log', _mock_log):
        # Only mock _encode if flash-attn is not available
        if not FLASH_ATTN_AVAILABLE:
            with patch.object(BiEncoderContrastiveModel, '_encode', _mock_encode):
                yield
        else:
            yield


def create_mock_batch(mock_dataset, batch_size=5, pad_token_id=0, cls_token_id=1, cls_value=-2, mask_value=-1):
    """
    Create a mock batch from the dataset with proper padding and CLS tokens.
    
    Args:
        mock_dataset: Mock dataset to sample from
        batch_size: Number of samples in the batch
        pad_token_id: Token ID for padding
        cls_token_id: Token ID for CLS token
        cls_value: Value for CLS token
        mask_value: Value for padding
    
    Returns:
        tuple: (tokens_tensor, values_tensor, padding_mask)
    """
    # Get samples from the mock dataset
    samples = [mock_dataset[i] for i in range(batch_size)]
    
    # Process each sample and add CLS token
    batch_tokens = []
    batch_values = []
    max_seq_len = 0
    
    for sample in samples:
        tokens = np.concatenate([[cls_token_id], sample['tokens']])  # Add CLS token
        values = np.concatenate([[cls_value], sample['values']])  # Add CLS value
        batch_tokens.append(tokens)
        batch_values.append(values)
        max_seq_len = max(max_seq_len, len(tokens))
    
    # Pad sequences to the same length using numpy arrays instead of lists of ndarrays
    padded_tokens = np.stack([
        np.concatenate([tokens, np.full(max_seq_len - len(tokens), pad_token_id, dtype=np.int64)])
        for tokens in batch_tokens
    ])
    padded_values = np.stack([
        np.concatenate([values, np.full(max_seq_len - len(values), mask_value, dtype=np.float32)])
        for values in batch_values
    ])
    
    # Convert to tensors
    tokens_tensor = torch.tensor(padded_tokens, dtype=torch.long)
    values_tensor = torch.tensor(padded_values, dtype=torch.float32)
    
    # Create padding mask
    padding_mask = torch.zeros_like(tokens_tensor, dtype=torch.bool)
    
    return tokens_tensor, values_tensor, padding_mask


class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.PAD_TOKEN = 0
        self.CLS_TOKEN = 1
        self.gene_mapping = {
            '<pad>': 0,
            '<cls>': 1,
            **{f'GENE_{i}': i + 2 for i in range(vocab_size - 2)}
        }
    
    def encode(self, gene_list):
        """Mock encode method that returns random token IDs"""
        return np.random.randint(2, self.vocab_size, size=len(gene_list))

# Mock GeneIdTokenizer
class MockGeneIdTokenizer(MockTokenizer):
    """Mock GeneIdTokenizer that inherits from MockTokenizer"""
    pass


class MockDataset:
    """Mock dataset for testing"""
    def __init__(self, num_samples=10, vocab_size=1000, max_tokens=50):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.collection = MagicMock()
        self.collection.storage_idx = np.arange(num_samples)
        self.collection._cached_obs = {}
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random tokens and values
        seq_len = np.random.randint(10, self.max_tokens)
        tokens = np.random.randint(2, self.vocab_size, size=seq_len)
        values = np.random.exponential(1.0, size=seq_len).astype(np.float32)
        
        return {
            'tokens': tokens,
            'values': values,
            'dataset_id': np.array([idx % 2]),  # Simulate 2 datasets
            'panel': np.random.randint(2, self.vocab_size, size=10),  # Mock panel
            'panel_name': f'panel_{idx % 3}',
        }


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing"""
    return {
        'flash_attention': False,  # Set to False to use regular transformer
        'dim_model': 64,
        'num_head': 4,
        'dim_hid': 128,
        'nlayers': 2,
        'dropout': 0.1,
        'decoder_head': False,
        'mask_padding': False,
        'input_encoding': 'rank_encoding',
        'pe_max_len': 1000,
        'mlm_loss_weight': 0.0,
        'cont_loss_weight': 1.0,
        'contrastive_loss': 'multiclass',
        'logit_scale_init_value': 1.0,
        'per_view_normalization': False,
        'random_split': False,
        'loss_switch_step': 1000,
        'values_only_sanity_check': False,
        'data_loading_speed_sanity_check': False,
        'projection_dim': None,
        'training': {
            'masking_rate': 0.0,
            'lr': 1e-3,
            'weight_decay': 0.0,
            'optimizer_class': 'AdamW',
            'scheduler': 'warmup',
            'warmup': 100,
            'max_steps': 1000,
            'min_lr': 0.0,
        }
    }


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer"""
    # Create a simple mock gene mapping for testing
    gene_mapping = {
        '<pad>': 0,
        '<cls>': 1,
        **{f'GENE_{i}': i + 2 for i in range(98)}  # 100 total tokens
    }
    return GeneIdTokenizer(gene_mapping)


@pytest.fixture
def mock_dataset():
    """Create a mock dataset"""
    return MockDataset(num_samples=20, vocab_size=100, max_tokens=30)



def test_model_initialization(mock_config, mock_tokenizer, device):
    """Test that the model can be initialized with mock config"""
    model = BiEncoderContrastiveModel(
        config=mock_config,
        pad_token_id=0,
        cls_token_id=1,
        vocab_size=100,
        world_size=1,
        val_loader_names=[]
    )
    
    # Move model to device
    model = model.to(device)
    
    assert model is not None
    assert model.vocab_size == 100
    assert model.dim_model == 64
    assert model.num_head == 4
    assert model.device.type == device.type


def test_model_forward_pass(mock_config, mock_tokenizer, mock_dataset, device):
    """Test that the model can perform a forward pass with batch size 5"""
    model = BiEncoderContrastiveModel(
        config=mock_config,
        pad_token_id=0,
        cls_token_id=1,
        vocab_size=100,
        world_size=1,
        val_loader_names=[]
    )
    
    # Move model to device
    model = model.to(device)
    
    # Create mock batch using the helper function
    batch_size = 5
    tokens_tensor, values_tensor, padding_mask = create_mock_batch(
        mock_dataset, 
        batch_size=batch_size,
        pad_token_id=0,
        cls_token_id=1,
        cls_value=-2,
        mask_value=-1
    )
    
    # Move tensors to device
    tokens_tensor = tokens_tensor.to(device)
    values_tensor = values_tensor.to(device)
    padding_mask = padding_mask.to(device)
    
    # Forward pass
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            pred, embs, cell_embs = model(tokens_tensor, values_tensor, padding_mask)
    
    # Check output shapes
    assert cell_embs.shape == (batch_size, mock_config['dim_model'])
    assert embs.shape[0] == batch_size  # batch size
    assert embs.shape[1] == tokens_tensor.shape[1]  # sequence length
    assert embs.shape[2] == mock_config['dim_model']  # embedding dimension
    # Check that outputs are on the correct device
    assert cell_embs.device.type == device.type
    assert embs.device.type == device.type


def test_training_step(mock_config, mock_tokenizer, mock_dataset, device):
    """Test that the model can perform a training step"""
    model = BiEncoderContrastiveModel(
        config=mock_config,
        pad_token_id=0,
        cls_token_id=1,
        vocab_size=100,
        world_size=1,
        val_loader_names=["val_test"]
    )
    
    # Move model to device
    model = model.to(device)
    
    # Create mock batch data
    batch_size = 8
    seq_len = 20
    
    panel = torch.randint(2, 100, (1, 10))
    panel = panel.repeat(batch_size, 1)
    
    # Create mock batch with paired data
    batch = {
        'tokens_1': torch.randint(2, 100, (batch_size, seq_len)).to(device),
        'values_1': torch.randn(batch_size, seq_len).to(device),
        'panel_1': panel.to(device),
        'tokens_2': torch.randint(2, 100, (batch_size, seq_len)).to(device),
        'values_2': torch.randn(batch_size, seq_len).to(device),
        'panel_2': panel.to(device),
        'dataset_id': torch.randint(0, 2, (batch_size,)).to(device),
        'panel_name': 'test_panel'
    }

    # Test training step
    model.train()
    with patch.object(model, "log") as mock_log:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss = model.training_step(batch, batch_idx=0)
        
        call_args_list = [call.args[0] for call in mock_log.call_args_list]
        assert any(call == 'train/loss' for call in call_args_list)
        assert any(call == 'train/loss_cont' for call in call_args_list)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert not torch.isnan(loss)
        assert loss.device.type == device.type


def test_predict_step(mock_config, mock_tokenizer, mock_dataset, device):
    """Test that the model can perform a predict step"""
    model = BiEncoderContrastiveModel(
        config=mock_config,
        pad_token_id=0,
        cls_token_id=1,
        vocab_size=100,
        world_size=1,
        val_loader_names=["val_test"]
    )
    
    # Move model to device
    model = model.to(device)
    
    # Create mock batch data for prediction
    batch_size = 8
    seq_len = 20
    
    # Create mock batch for prediction (single view, not paired)
    batch = {
        'tokens': torch.randint(2, 100, (batch_size, seq_len)).to(device),
        'values': torch.randn(batch_size, seq_len).to(device),
        'dataset_id': torch.randint(0, 2, (batch_size,)).to(device),
        'panel_name': 'test_panel'
    }
    
    # Test predict step
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            result = model.predict_step(batch, batch_idx=0)
    
    # Check that we get the expected keys
    expected_keys = ['pred', 'cls_cell_emb', 'mean_cell_emb', 'context_sizes']
    for key in expected_keys:
        assert key in result
    
    # Check output shapes
    assert result['cls_cell_emb'].shape == (batch_size, mock_config['dim_model'])
    assert result['mean_cell_emb'].shape == (batch_size, mock_config['dim_model'])
    assert isinstance(result['context_sizes'], tuple)
    assert len(result['context_sizes']) == 2  # (context_size, nonzero_cnt)
    
    # Check that pred is None when decoder_head is False
    assert result['pred'] is None
    
    # Check that outputs are on the correct device
    assert result['cls_cell_emb'].device.type == device.type
    assert result['mean_cell_emb'].device.type == device.type


from unittest.mock import patch

def test_validation_step(mock_config, mock_tokenizer, mock_dataset, device):
    """Test that the model can perform a validation step and calls model.log appropriately"""
    model = BiEncoderContrastiveModel(
        config=mock_config,
        pad_token_id=0,
        cls_token_id=1,
        vocab_size=100,
        world_size=1,
        val_loader_names=['test_val']
    )
    
    # Move model to device
    model = model.to(device)
    
    # Create mock batch data
    batch_size = 8
    seq_len = 20
    
    panel = torch.randint(2, 100, (1, 10))
    panel = panel.repeat(batch_size, 1)
    
    batch = {
        'tokens_1': torch.randint(2, 100, (batch_size, seq_len)).to(device),
        'values_1': torch.randn(batch_size, seq_len).to(device),
        'panel_1': panel.to(device),
        'tokens_2': torch.randint(2, 100, (batch_size, seq_len)).to(device),
        'values_2': torch.randn(batch_size, seq_len).to(device),
        'panel_2': panel.to(device),
        'dataset_id': torch.randint(0, 2, (batch_size,)).to(device),
        'panel_name': 'test_panel'
    }
    
    # Test validation step and check model.log calls
    model.eval()
    with torch.no_grad(), patch.object(model, "log") as mock_log:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            model.validation_step(batch, batch_idx=0, dataloader_idx=0)

        call_args_list = [call.args[0] for call in mock_log.call_args_list]
        assert any(call == 'val/test_val/loss' for call in call_args_list)
        assert any(call == 'val/test_val/loss_cont' for call in call_args_list)
        

@pytest.mark.parametrize("batch_size,max_tokens,gene_sampling_strategy", [
    (8, 20, 'random'),
    (8, 20, 'top'),
    (8, 20, 'random-nonzero'),
    (8, 20, 'top-nonzero'),
    (1, 20, 'top-nonzero'),
    (8, 2, 'top-nonzero'),
    (8, 1, 'top-nonzero'),
])
def test_predict_step_with_lamin_dataloader(mock_config, device, adata, tokenizer, batch_size, max_tokens, gene_sampling_strategy):
    """Test predict_step using InMemoryTokenizedDataset and CustomCollate from lamin_dataloader with different parameters"""
    
    # Create model
    model = BiEncoderContrastiveModel(
        config=mock_config,
        pad_token_id=0,
        cls_token_id=1,
        vocab_size=len(tokenizer.gene_mapping),
        world_size=1,
        val_loader_names=[]
    )
    model = model.to(device)
    model.eval()
    
    # Create InMemoryTokenizedDataset
    test_dataset = InMemoryTokenizedDataset(
        adata.copy(),
        tokenizer,
        normalization='raw',
        var_column=None
    )
    
    # Create CustomCollate with parameterized max_tokens
    collate_fn = LaminCustomCollate(
        tokenizer.PAD_TOKEN,
        max_tokens=max_tokens,
        gene_sampling_strategy=gene_sampling_strategy
    )
    
    # Create DataLoader with parameterized batch_size
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False
    )
    
    # Test predict_step with real data pipeline
    all_outputs = []
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value 
                        for key, value in batch.items()}
                
                # Call predict_step
                output = model.predict_step(batch, batch_idx)
                all_outputs.append(output)
                
                # Verify output structure
                expected_keys = ['pred', 'cls_cell_emb', 'mean_cell_emb']
                for key in expected_keys:
                    assert key in output, f"Missing key: {key}"
                
                # Verify output shapes
                actual_batch_size = batch['tokens'].shape[0]
                assert output['cls_cell_emb'].shape == (actual_batch_size, mock_config['dim_model'])
                assert output['mean_cell_emb'].shape == (actual_batch_size, mock_config['dim_model'])
                
                # Only test first few batches to keep test fast
                if batch_idx >= 2:
                    break
    
    # Verify we got some outputs
    assert len(all_outputs) > 0, f"No outputs generated for batch_size={batch_size}, max_tokens={max_tokens}"
    
    # Test that we can concatenate all cell embeddings
    all_cls_embs = torch.cat([output['cls_cell_emb'] for output in all_outputs], dim=0)
    all_mean_embs = torch.cat([output['mean_cell_emb'] for output in all_outputs], dim=0)
    
    # Verify final shapes
    total_cells = sum(batch['tokens'].shape[0] for batch in list(dataloader)[:3])
    assert all_cls_embs.shape[0] == total_cells
    assert all_mean_embs.shape[0] == total_cells
    assert all_cls_embs.shape[1] == mock_config['dim_model']
    assert all_mean_embs.shape[1] == mock_config['dim_model']

if __name__ == "__main__":
    pytest.main([__file__])
