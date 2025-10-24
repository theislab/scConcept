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
from lamin_dataloader.dataset import GeneIdTokenizer


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
        


# @patch('concept.data.datamodules.MappedCollection')
# @patch('concept.data.datamodules.TokenizedDataset')
# def test_datamodule_initialization(mock_dataset_class, mock_collection, mock_tokenizer):
#     """Test that the datamodule can be initialized with mock data"""
#     # Mock the dataset and collection
#     mock_dataset_instance = MockDataset()
#     mock_dataset_class.return_value = mock_dataset_instance
#     mock_collection.return_value = MagicMock()
    
#     # Create temporary directory for panels
#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Create a mock panel file
#         panel_file = Path(temp_dir) / "test_panel.csv"
#         panel_df = pd.DataFrame({
#             'Ensembl_ID': [f'GENE_{i}' for i in range(10)]
#         })
#         panel_df.to_csv(panel_file, index=False)
        
#         # Initialize datamodule
#         datamodule = MappedCollectionDataModule(
#             dataset_path=temp_dir,
#             split={'train': ['test.h5ad'], 'val': ['test_val.h5ad']},
#             panels_path=temp_dir,
#             tokenizer=mock_tokenizer,
#             columns=['test_col'],
#             precomp_embs_key=None,
#             normalization='log1p',
#             gene_sampling_strategy='random-nonzero',
#             model_speed_sanity_check=False,
#             dataset_kwargs={},
#             dataloader_kwargs={
#                 'batch_size': 4,
#                 'shuffle': True,
#                 'drop_last': True,
#                 'num_workers': 0,
#                 'num_samples': None,
#                 'within_group_sampling': None
#             },
#             val_loader_names=[]
#         )
        
#         assert datamodule is not None
#         assert datamodule.tokenizer == mock_tokenizer


# def test_collate_functionality(mock_tokenizer):
#     """Test that the collate function works with mock data"""
#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Create a mock panel file
#         panel_file = Path(temp_dir) / "test_panel.csv"
#         panel_df = pd.DataFrame({
#             'Ensembl_ID': [f'GENE_{i}' for i in range(10)]
#         })
#         panel_df.to_csv(panel_file, index=False)
        
#         # Initialize collate function
#         collate_fn = CustomCollate(
#             tokenizer=mock_tokenizer,
#             panels_path=temp_dir,
#             max_tokens=50,
#             min_tokens=5,
#             split_input=True,
#             variable_size=False,
#             gene_sampling_strategy='random-nonzero',
#             panel_selection='random',
#             panel_selection_mixed_prob=1.0,
#             panel_filter_regex='.*',
#             panel_size_min=None,
#             panel_size_max=None,
#             panel_overlap=False,
#             anchor_panel_size=None,
#             anchor_max_tokens=None,
#             panel_max_drop_rate=None,
#             feature_max_drop_rate=None,
#             model_speed_sanity_check=False
#         )
        
#         # Create mock batch
#         batch = [
#             {
#                 'tokens': np.random.randint(2, 100, size=20),
#                 'values': np.random.exponential(1.0, size=20).astype(np.float32),
#                 'dataset_id': np.array([0]),
#                 'panel': np.random.randint(2, 100, size=10),
#                 'panel_name': 'test_panel'
#             }
#             for _ in range(4)
#         ]
        
#         # Test collate function
#         collated_batch = collate_fn(batch)
        
#         # Check that we get the expected keys
#         expected_keys = ['tokens_1', 'values_1', 'panel_1', 'tokens_2', 'values_2', 'panel_2', 'dataset_id', 'panel_name']
#         for key in expected_keys:
#             assert key in collated_batch
        
#         # Check tensor shapes
#         assert collated_batch['tokens_1'].shape[0] == 4  # batch size
#         assert collated_batch['values_1'].shape[0] == 4  # batch size
#         assert collated_batch['tokens_2'].shape[0] == 4  # batch size
#         assert collated_batch['values_2'].shape[0] == 4  # batch size


if __name__ == "__main__":
    pytest.main([__file__])
