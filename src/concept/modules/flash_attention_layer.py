import torch
from torch import Tensor
from typing import Optional
from flash_attn.modules.mha import (
    MHA,
)  # https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mha.py

import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class FlashTransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    The class is modified from torch.nn.TransformerEncoderLayer to support the
    FlashAttention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """

    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        use_flash_attn=True,
        device=None,
        dtype=None,
        norm_scheme="post",  # "pre" or "post"
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if use_flash_attn:
            logger.info("Using FlashAttention")

        self.self_attn = MHA(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
            **factory_kwargs,
        )
        # Version compatibility workaround
        if not hasattr(self.self_attn, "batch_first"):
            self.self_attn.batch_first = batch_first
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if self.norm_scheme not in ["pre", "post"]:
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if src_mask is not None:
            raise ValueError("FlashTransformerEncoderLayer does not support src_mask")

        if key_padding_mask is not None:
            key_padding_mask = ~key_padding_mask

        if self.norm_scheme == "pre":
            src = self.norm1(src)
            src2 = self.self_attn(src, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, key_padding_mask=key_padding_mask)
            src = src + self.dropout1(src2)
            src = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
        else:
            src2 = self.self_attn(src, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, key_padding_mask=key_padding_mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src


class FlashTransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    The class is modified from torch.nn.TransformerDecoderLayer to support FlashAttention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``True``.
        use_flash_attn: whether to use flash attention (default=True).
        norm_scheme: normalization scheme, "pre" or "post" (default="post").

    Examples::
        >>> decoder_layer = FlashTransformerDecoderLayer(d_model=512, nhead=8)
        >>> tgt = torch.rand(32, 10, 512)
        >>> memory = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        use_flash_attn=True,
        device=None,
        dtype=None,
        norm_scheme="post",  # "pre" or "post"
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if use_flash_attn:
            logger.info("Using FlashAttention")
        # Self-attention
        self.self_attn = MHA(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
            **factory_kwargs,
        )

        # Cross-attention
        self.multihead_attn = MHA(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
            cross_attn=True,
            **factory_kwargs,
        )

        # Version compatibility workaround
        if not hasattr(self.self_attn, "batch_first"):
            self.self_attn.batch_first = batch_first
        if not hasattr(self.multihead_attn, "batch_first"):
            self.multihead_attn.batch_first = batch_first

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if self.norm_scheme not in ["pre", "post"]:
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            cu_seqlens: cumulative sequence lengths (optional, for variable length).
            max_seqlen: maximum sequence length (optional, for variable length).

        Shape:
            see the docs in Transformer class.
        """
        if tgt_mask is not None:
            raise ValueError("FlashTransformerDecoderLayer does not support tgt_mask")

        if memory_mask is not None:
            raise ValueError("FlashTransformerDecoderLayer does not support memory_mask")

        # Flash attention expects inverted padding mask
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = ~tgt_key_padding_mask

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = ~memory_key_padding_mask

        if self.norm_scheme == "pre":
            # Self-attention block
            tgt = self.norm1(tgt)
            tgt2 = self.self_attn(
                tgt, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, key_padding_mask=tgt_key_padding_mask
            )
            tgt = tgt + self.dropout1(tgt2)

            # Cross-attention block
            tgt = self.norm2(tgt)
            tgt2 = self.multihead_attn(tgt, x_kv=memory, key_padding_mask=memory_key_padding_mask)
            tgt = tgt + self.dropout2(tgt2)

            # Feedforward block
            tgt = self.norm3(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
        else:
            # Self-attention block
            tgt2 = self.self_attn(
                tgt, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, key_padding_mask=tgt_key_padding_mask
            )
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

            # Cross-attention block
            tgt2 = self.multihead_attn(tgt, x_kv=memory, key_padding_mask=memory_key_padding_mask)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

            # Feedforward block
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)

        return tgt
