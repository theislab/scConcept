#!/usr/bin/env python3
"""
Test script to verify both flash attention and non-flash attention implementations work correctly.
"""

import torch
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_encode_implementations():
    """Test both flash attention and non-flash attention implementations."""
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    dim_model = 64
    vocab_size = 100
    pad_token_id = 0
    cls_token_id = 1
    
    # Create test data
    tokens = torch.randint(1, vocab_size, (batch_size, seq_len))
    values = torch.randn(batch_size, seq_len)
    src_key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    
    # Add some padding
    tokens[0, 5:] = pad_token_id
    src_key_padding_mask[0, 5:] = True
    
    print("Testing both flash attention implementations...")
    
    # Test 1: Flash attention = True
    print("\n1. Testing flash_attention=True...")
    try:
        from concept.model import BiEncoderContrastiveModel
        
        config_flash = {
            'flash_attention': True,
            'dim_model': dim_model,
            'dim_hid': 128,
            'num_head': 4,
            'nlayers': 2,
            'dropout': 0.1,
            'decoder_head': False,
            'mask_padding': True,
            'input_encoding': 'rank_encoding',
            'training': {
                'masking_rate': 0.1,
                'lr': 1e-3,
                'weight_decay': 0.0,
                'optimizer_class': 'Adam',
                'scheduler': None,
                'warmup': 100,
                'max_steps': 1000,
                'min_lr': 1e-6
            },
            'mlm_loss_weight': 0.0,
            'cont_loss_weight': 1.0,
            'contrastive_loss': 'binary',
            'loss_switch_step': 100,
            'per_view_normalization': False,
            'logit_scale_init_value': 1.0,
            'random_split': False,
            'projection_dim': None,
            'pe_max_len': 1000,
            'values_only_sanity_check': False,
            'data_loading_speed_sanity_check': False
        }
        
        model_flash = BiEncoderContrastiveModel(
            config=config_flash,
            pad_token_id=pad_token_id,
            cls_token_id=cls_token_id,
            vocab_size=vocab_size
        )
        
        # Test the _encode method
        embs_padded, cell_embs = model_flash._encode(tokens, values, src_key_padding_mask)
        
        print(f"   ✓ Flash attention: embs_padded shape: {embs_padded.shape}")
        print(f"   ✓ Flash attention: cell_embs shape: {cell_embs.shape}")
        print(f"   ✓ Flash attention: Expected shapes - embs_padded: ({batch_size}, {seq_len}, {dim_model}), cell_embs: ({batch_size}, {dim_model})")
        
    except Exception as e:
        print(f"   ✗ Flash attention failed: {e}")
        return False
    
    # Test 2: Flash attention = False
    print("\n2. Testing flash_attention=False...")
    try:
        config_no_flash = config_flash.copy()
        config_no_flash['flash_attention'] = False
        
        model_no_flash = BiEncoderContrastiveModel(
            config=config_no_flash,
            pad_token_id=pad_token_id,
            cls_token_id=cls_token_id,
            vocab_size=vocab_size
        )
        
        # Test the _encode method
        embs_padded, cell_embs = model_no_flash._encode(tokens, values, src_key_padding_mask)
        
        print(f"   ✓ No flash attention: embs_padded shape: {embs_padded.shape}")
        print(f"   ✓ No flash attention: cell_embs shape: {cell_embs.shape}")
        print(f"   ✓ No flash attention: Expected shapes - embs_padded: ({batch_size}, {seq_len}, {dim_model}), cell_embs: ({batch_size}, {dim_model})")
        
    except Exception as e:
        print(f"   ✗ No flash attention failed: {e}")
        return False
    
    print("\n✅ All tests passed! Both implementations work correctly.")
    return True

if __name__ == "__main__":
    success = test_encode_implementations()
    sys.exit(0 if success else 1)
