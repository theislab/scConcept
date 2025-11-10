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
import numpy as np
import sys
from unittest.mock import patch, MagicMock

# Check if flash-attn is installed
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

print(f"\n\nFLASH_ATTN_AVAILABLE: {FLASH_ATTN_AVAILABLE}")

# Mock flash_attn and its submodules only if flash-attn is not installed
if not FLASH_ATTN_AVAILABLE:
    print("Mocking flash_attn")
    sys.modules['flash_attn'] = MagicMock()
    sys.modules['flash_attn.modules'] = MagicMock()
    sys.modules['flash_attn.modules.mha'] = MagicMock()

# Now we can import the modules
from concept.model import BiEncoderContrastiveModel
from lamin_dataloader.dataset import TokenizedDataset
from lamin_dataloader.collections import InMemoryCollection
from lamin_dataloader.dataset import BaseCollate
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


def test_model_initialization(mock_config, device):
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


def test_training_step(mock_config, device):
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


def test_validation_step(mock_config, device):
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
        

def test_predict_step(mock_config, device):
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



@pytest.mark.parametrize("batch_size,max_tokens,gene_sampling_strategy", [
    (8, 20, 'random'),
    (8, 20, 'top'),
    (8, 20, 'random-nonzero'),
    (8, 20, 'top-nonzero'),
    (1, 20, 'top-nonzero'),
    (8, 2, 'top-nonzero'),
    (8, 1, 'top-nonzero'),
])
def test_predict_step_with_lamin_dataloader(
    mock_config, 
    device, 
    adata, 
    tokenizer, 
    batch_size, 
    max_tokens, 
    gene_sampling_strategy):
    """Test predict_step using InMemory TokenizedDataset and 
    BaseCollate from lamin_dataloader with different parameters"""
    
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
    
    # Create InMemory TokenizedDataset
    test_dataset = TokenizedDataset(
        collection=InMemoryCollection([adata]),
        tokenizer=tokenizer,
        normalization='raw'
    )
    
    # Create BaseCollate with parameterized max_tokens
    collate_fn = BaseCollate(
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


@pytest.mark.skipif(not FLASH_ATTN_AVAILABLE, reason="flash_att is not available")
def test_scConcept_integration(adata):
    """Integration test for scConcept class with real HuggingFace model"""
    from concept.scConcept import scConcept    
    
    # Initialize scConcept
    sc_concept = scConcept()
    
    # Test model loading
    sc_concept.load_config_and_model(model_name="Corpus-30M")
    
    # Verify model is loaded correctly
    assert sc_concept.model is not None
    assert sc_concept.tokenizer is not None
    assert sc_concept.device is not None
    assert sc_concept.cfg is not None
    
    # Test embedding extraction from AnnData
    batch_size = 4
    
    result = sc_concept.extract_embeddings(
        adata=adata,
        batch_size=batch_size,
    )
    
    # Verify result structure
    expected_keys = ['cls_cell_emb', 'mean_cell_emb']
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"
    
    # Verify embedding shapes
    n_cells = adata.shape[0]
    assert result['cls_cell_emb'].shape == (n_cells, sc_concept.model.dim_model)
    assert result['mean_cell_emb'].shape == (n_cells, sc_concept.model.dim_model)
    
    # Verify embeddings are numpy arrays
    assert isinstance(result['cls_cell_emb'], np.ndarray)
    assert isinstance(result['mean_cell_emb'], np.ndarray)
    
    # Verify embeddings are not all zeros or NaNs
    assert not np.all(result['cls_cell_emb'] == 0)
    assert not np.all(result['mean_cell_emb'] == 0)
    assert not np.any(np.isnan(result['cls_cell_emb']))
    assert not np.any(np.isnan(result['mean_cell_emb']))
    
    # Test training with pre-trained model (resuming/fine-tuning)
    # Store initial model state for comparison
    initial_state = {}
    for name, param in sc_concept.model.named_parameters():
        if param.requires_grad:
            initial_state[name] = param.data.clone()
    
    # Train for a few steps to verify training works
    # Use small max_steps and batch_size to keep test fast
    training_max_steps = 3
    training_batch_size = 8
    
    # Verify model is in eval mode before training
    assert not sc_concept.model.training, "Model should be in eval mode after loading"
    
    # Run training
    sc_concept.train(
        adata_list=adata,
        max_steps=training_max_steps,
        batch_size=training_batch_size
    )
    
    # Verify model is still valid after training
    assert sc_concept.model is not None
    
    # Verify that model parameters have changed (training occurred)
    # Check at least one parameter has changed
    parameters_changed = False
    for name, param in sc_concept.model.named_parameters():
        if param.requires_grad and name in initial_state:
            # Ensure both tensors are on the same device for comparison
            initial_param = initial_state[name].to(param.data.device)
            if not torch.equal(param.data, initial_param):
                parameters_changed = True
                break
    
    assert parameters_changed, "Model parameters should have changed after training"



def test_anndatamodule_integration(adata, tokenizer, device, tmp_path):
    """Integration test for AnnDataModule with AnnData objects
    
    Tests that AnnDataModule can:
    - Initialize with AnnData objects
    - Create train and test dataloaders
    - Iterate through dataloaders and produce valid batches
    - Handle collate functions properly
    """
    from concept.data.datamodules import AnnDataModule
    
    # Create a simple panels file for testing
    panels_dir = tmp_path / "panels"
    panels_dir.mkdir()
    panel_file = panels_dir / "test_panel.csv"
    
    # Create a panel with some genes from adata
    import pandas as pd
    panel_genes = adata.var_names[:5].tolist()  # Use first 5 genes
    panel_df = pd.DataFrame({'Ensembl_ID': panel_genes})
    panel_df.to_csv(panel_file, index=False)
    
    # Configuration for AnnDataModule
    split = {
        'train': [adata],  # List of AnnData objects
        'test': [adata]
    }
    
    columns = []  # No obs columns needed for basic test
    
    dataset_kwargs = {
        'train': {
            'max_tokens': 20,
            'variable_size': False,
            'panel_selection': 'mixed',
            'panel_selection_mixed_prob': 0.5,
            'panel_filter_regex': '.*',
            'panel_size_min': 5,
            'panel_size_max': 10,
            'panel_max_drop_rate': 0.1,
            'feature_max_drop_rate': 0.1,
        },
        'test': {
            'max_tokens': 20,
        }
    }
    
    dataloader_kwargs = {
        'train': {
            'batch_size': 4,
            'shuffle': True,
            'drop_last': True,
            'num_samples': None,
            'num_workers': 0,
            'within_group_sampling': 'dataset',
        },
        'test': {
            'batch_size': 4,
            'shuffle': False,
            'drop_last': False,
            'num_workers': 0,
        }
    }
    
    # Initialize AnnDataModule
    datamodule = AnnDataModule(
        split=split,
        panels_path=str(panels_dir),
        tokenizer=tokenizer,
        columns=columns,
        precomp_embs_key=None,
        normalization='raw',
        gene_sampling_strategy='top-nonzero',
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
        val_loader_names=[]
    )
    
    # Test train dataloader
    train_loader = datamodule.train_dataloader()
    assert train_loader is not None
    
    # Iterate through train loader
    train_batches = []
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value 
                for key, value in batch.items()}
        
        # Verify batch structure
        assert 'tokens_1' in batch
        assert 'values_1' in batch
        assert 'tokens_2' in batch
        assert 'values_2' in batch
        assert 'panel_1' in batch
        assert 'panel_2' in batch
        
        # Verify batch shapes
        batch_size = batch['tokens_1'].shape[0]
        assert batch_size <= 4  # Max batch size
        assert batch['values_1'].shape == batch['tokens_1'].shape
        assert batch['values_2'].shape == batch['tokens_2'].shape
        
        # Verify tensors are on correct device
        assert batch['tokens_1'].device.type == device.type
        assert batch['values_1'].device.type == device.type
        
        train_batches.append(batch)
        
        # Only test first 2 batches to keep test fast
        if batch_idx >= 1:
            break
    
    assert len(train_batches) > 0, "No train batches generated"
    
    # Test test dataloader
    test_loader = datamodule.test_dataloader()
    assert test_loader is not None
    
    # Iterate through test loader
    test_batches = []
    for batch_idx, batch in enumerate(test_loader):
        # Move batch to device
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value 
                for key, value in batch.items()}
        
        # Verify batch structure (test loader doesn't split input)
        assert 'tokens' in batch
        assert 'values' in batch
        
        # Verify batch shapes
        batch_size = batch['tokens'].shape[0]
        assert batch_size <= 4  # Max batch size
        assert batch['values'].shape == batch['tokens'].shape
        
        # Verify tensors are on correct device
        assert batch['tokens'].device.type == device.type
        assert batch['values'].device.type == device.type
        
        test_batches.append(batch)
        
        # Only test first 2 batches to keep test fast
        if batch_idx >= 1:
            break
    
    assert len(test_batches) > 0, "No test batches generated"
    
    print(f"\n✓ AnnDataModule test passed:")
    print(f"  - Generated {len(train_batches)} train batches")
    print(f"  - Generated {len(test_batches)} test batches")
    print(f"  - Train batch shape: {train_batches[0]['tokens_1'].shape}")
    print(f"  - Test batch shape: {test_batches[0]['tokens'].shape}")


if __name__ == "__main__":
    pytest.main([__file__])
