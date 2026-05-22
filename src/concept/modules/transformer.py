"""
This file contains TransformerEncoder, TransformerDecoder and Transformer
copied almost verbatim from the PyTorch codebase. The only change is that
the "fast path" logic in the TransformerEncoder is removed. And the src/key/memory
padding mask is removed.

"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, ModuleList
import copy
from typing import Optional

# We use this for exact parity with the PyTorch implementation, having the same init
# for every layer might not be necessary.


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[Tensor] = None,
        key_padding_mask=None,
        is_causal=False,
    ):
        output = src
        for mod in self.layers:
            output = mod(
                output,
                mask,
                is_causal=is_causal,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                key_padding_mask=key_padding_mask,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output
