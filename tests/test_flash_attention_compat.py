import copy
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from concept.model import ContrastiveModel
from concept.modules.flash_attention_layer import FlashTransformerEncoderLayer


FLASH_ATTN_AVAILABLE = importlib.util.find_spec("flash_attn") is not None


def _model_config(*, flash_attention: bool, decoder_head: bool = False):
    return {
        "flash_attention": flash_attention,
        "dim_gene_embs": 16,
        "dim_model": 16,
        "num_head": 4,
        "dim_hid": 32,
        "nlayers": 2,
        "dropout": 0.0,
        "decoder_head": decoder_head,
        "mask_value": -1,
        "cls_value": -2,
        "mask_padding": False,
        "input_encoding": "rank_encoding",
        "pe_max_len": 32,
        "mlm_loss_weight": 0.0,
        "cont_loss_weight": 1.0,
        "contrastive_loss": "multiclass",
        "logit_scale_init_value": 1.0,
        "per_view_normalization": False,
        "random_split": False,
        "loss_switch_step": 1000,
        "values_only_sanity_check": False,
        "data_loading_speed_sanity_check": False,
        "projection_dim": None,
        "training": {
            "masking_rate": 0.0,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "optimizer_class": "AdamW",
            "scheduler": None,
            "freeze_pretrained_vocabulary": None,
            "use_learnable_embs_freq": None,
            "warmup": 0,
            "max_steps": 1,
            "min_lr": 0.0,
        },
    }


def _build_model(*, flash_attention: bool, decoder_head: bool = False, device="cpu"):
    model = ContrastiveModel(
        config=_model_config(flash_attention=flash_attention, decoder_head=decoder_head),
        pad_token_id=0,
        cls_token_id=1,
        vocab_sizes={"hsapiens": 32},
        world_size=1,
        val_loader_names=[],
    ).to(device)
    model.eval()
    model.stage = "predict"
    model.set_active_species("hsapiens")
    return model


def test_attention_backends_expose_same_parameter_schema():
    if not FLASH_ATTN_AVAILABLE:
        pytest.skip("flash_attn is not installed")

    flash_layer = FlashTransformerEncoderLayer(
        d_model=16,
        nhead=4,
        dim_feedforward=32,
        dropout=0.0,
        batch_first=True,
        use_flash_attn=True,
    )
    fallback_layer = FlashTransformerEncoderLayer(
        d_model=16,
        nhead=4,
        dim_feedforward=32,
        dropout=0.0,
        batch_first=True,
        use_flash_attn=False,
    )

    flash_state = flash_layer.self_attn.state_dict()
    fallback_state = fallback_layer.self_attn.state_dict()

    assert list(flash_state.keys()) == list(fallback_state.keys())
    for key in flash_state:
        assert flash_state[key].shape == fallback_state[key].shape


def test_encoder_layers_load_strictly_across_attention_backends():
    if not FLASH_ATTN_AVAILABLE:
        pytest.skip("flash_attn is not installed")

    flash_layer = FlashTransformerEncoderLayer(
        d_model=16,
        nhead=4,
        dim_feedforward=32,
        dropout=0.0,
        batch_first=True,
        use_flash_attn=True,
    )
    fallback_layer = FlashTransformerEncoderLayer(
        d_model=16,
        nhead=4,
        dim_feedforward=32,
        dropout=0.0,
        batch_first=True,
        use_flash_attn=False,
    )

    result = fallback_layer.load_state_dict(flash_layer.state_dict(), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_model_checkpoint_loads_between_flash_and_fallback_backends():
    if not FLASH_ATTN_AVAILABLE:
        pytest.skip("flash_attn is not installed")

    flash_model = _build_model(flash_attention=True)
    fallback_model = _build_model(flash_attention=False)

    result = fallback_model.load_state_dict(flash_model.state_dict(), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


@pytest.mark.skipif(not FLASH_ATTN_AVAILABLE or not torch.cuda.is_available(), reason="requires CUDA FlashAttention")
def test_model_eval_outputs_match_between_flash_and_fallback_backends():
    device = torch.device("cuda")
    flash_model = _build_model(flash_attention=True, decoder_head=True, device=device)
    fallback_model = _build_model(flash_attention=False, decoder_head=True, device=device)
    fallback_model.load_state_dict(flash_model.state_dict(), strict=True)

    tokens = torch.tensor(
        [
            [1, 7, 8, 9, 10],
            [1, 11, 12, 13, 0],
            [1, 14, 15, 0, 0],
        ],
        device=device,
    )
    values = torch.zeros(tokens.shape, device=device)
    seq_lengths = [5, 4, 3]

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        flash_preds, _, flash_cell_embs = flash_model(tokens, values, seq_lengths)
        fallback_preds, _, fallback_cell_embs = fallback_model(tokens, values, seq_lengths)

    non_pad_mask = tokens != flash_model.PAD_TOKEN_ID
    torch.testing.assert_close(flash_cell_embs, fallback_cell_embs, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(flash_preds[non_pad_mask], fallback_preds[non_pad_mask], rtol=5e-2, atol=5e-2)


def test_non_flash_import_and_instantiation_do_not_require_flash_attn(tmp_path):
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        "import builtins\n"
        "import importlib.util\n"
        "import sys\n"
        "_orig_import = builtins.__import__\n"
        "def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
        "    if name == 'flash_attn' or name.startswith('flash_attn.'):\n"
        "        raise ImportError('blocked for test')\n"
        "    return _orig_import(name, globals, locals, fromlist, level)\n"
        "builtins.__import__ = _blocked_import\n"
        "_orig_find_spec = importlib.util.find_spec\n"
        "def _blocked_find_spec(name, package=None):\n"
        "    if name == 'flash_attn' or name.startswith('flash_attn.'):\n"
        "        return None\n"
        "    return _orig_find_spec(name, package)\n"
        "importlib.util.find_spec = _blocked_find_spec\n"
        "for name in list(sys.modules):\n"
        "    if name == 'flash_attn' or name.startswith('flash_attn.'):\n"
        "        sys.modules.pop(name)\n",
        encoding="ascii",
    )

    env = copy.deepcopy(os.environ)
    env["PYTHONPATH"] = f"{tmp_path}{os.pathsep}{Path.cwd() / 'src'}"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from concept.model import ContrastiveModel\n"
                "m = ContrastiveModel("
                "config={'flash_attention': False, 'dim_gene_embs': 16, 'dim_model': 16, 'num_head': 4, "
                "'dim_hid': 32, 'nlayers': 1, 'dropout': 0.0, 'decoder_head': False, 'mask_value': -1, "
                "'cls_value': -2, 'mask_padding': False, 'input_encoding': 'rank_encoding', 'pe_max_len': 16, "
                "'mlm_loss_weight': 0.0, 'cont_loss_weight': 1.0, 'contrastive_loss': 'multiclass', "
                "'logit_scale_init_value': 1.0, 'per_view_normalization': False, 'random_split': False, "
                "'loss_switch_step': 1, 'values_only_sanity_check': False, 'data_loading_speed_sanity_check': False, "
                "'projection_dim': None, 'training': {'masking_rate': 0.0, 'lr': 1e-3, 'weight_decay': 0.0, "
                "'optimizer_class': 'AdamW', 'scheduler': None, 'freeze_pretrained_vocabulary': None, "
                "'use_learnable_embs_freq': None, 'warmup': 0, 'max_steps': 1, 'min_lr': 0.0}}, "
                "pad_token_id=0, cls_token_id=1, vocab_sizes={'hsapiens': 8}, world_size=1, val_loader_names=[])\n"
                "print(type(m.transformer_encoder.layers[0].self_attn).__name__)\n"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "FallbackMHA" in result.stdout
