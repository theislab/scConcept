from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FallbackMHA(nn.Module):
    """Subset of flash_attn.modules.mha.MHA with a compatible checkpoint schema."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        cross_attn=False,
        qkv_proj_bias=True,
        out_proj_bias=True,
        dropout=0.0,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        dwconv=False,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        use_alibi=False,
        window_size=(-1, -1),
        fused_bias_fc=False,
        use_flash_attn=False,
        return_residual=False,
        checkpointing=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if use_flash_attn:
            raise ValueError("FallbackMHA only supports use_flash_attn=False")
        if cross_attn:
            raise NotImplementedError("FallbackMHA only supports self-attention")
        if num_heads_kv not in (None, num_heads):
            raise NotImplementedError("FallbackMHA does not support grouped-query attention")
        if any(
            [
                dwconv,
                rotary_emb_dim != 0,
                use_alibi,
                window_size != (-1, -1),
                fused_bias_fc,
                return_residual,
                checkpointing,
                layer_idx is not None,
                rotary_emb_base != 10000.0,
                rotary_emb_scale_base is not None,
                rotary_emb_interleaved,
            ]
        ):
            raise NotImplementedError("FallbackMHA only implements the attention features used in this project")

        self.embed_dim = embed_dim
        self.cross_attn = cross_attn
        self.causal = causal
        self.layer_idx = layer_idx
        self.dwconv = dwconv
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.return_residual = return_residual
        self.checkpointing = checkpointing
        self.num_heads = num_heads
        self.num_heads_kv = num_heads if num_heads_kv is None else num_heads_kv
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.softmax_scale = softmax_scale
        self.dropout_p = dropout

        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        self.Wqkv = nn.Linear(embed_dim, qkv_dim, bias=qkv_proj_bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias, **factory_kwargs)

    def forward(
        self,
        x,
        x_kv=None,
        key_padding_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
        mixer_subset=None,
        inference_params=None,
        **kwargs,
    ):
        if x_kv is not None or mixer_subset is not None or inference_params is not None:
            raise NotImplementedError("FallbackMHA only supports plain self-attention")
        if cu_seqlens is not None or max_seqlen is not None:
            raise ValueError("FallbackMHA only supports padded inputs without cu_seqlens/max_seqlen")
        if kwargs:
            unexpected_args = ", ".join(sorted(kwargs))
            raise TypeError(f"FallbackMHA got unsupported arguments: {unexpected_args}")
        if x.dim() != 3:
            raise ValueError("FallbackMHA expects padded batch-first inputs of shape (batch, seq, dim)")

        batch_size, seqlen, _ = x.shape
        qkv = self.Wqkv(x).view(batch_size, seqlen, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch_size, seqlen):
                raise ValueError("key_padding_mask must have shape (batch, seq)")
            attn_mask = torch.zeros(
                batch_size,
                1,
                1,
                seqlen,
                dtype=q.dtype,
                device=q.device,
            )
            attn_mask = attn_mask.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(q.dtype).min)

        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=self.causal,
            scale=self.softmax_scale,
        )
        context = context.transpose(1, 2).contiguous().view(batch_size, seqlen, self.embed_dim)
        out = self.out_proj(context)
        return out if not self.return_residual else (out, x)
