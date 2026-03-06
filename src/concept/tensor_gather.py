import torch
import torch.distributed as dist
import logging
from torch import Tensor
from torch.distributed.nn.functional import all_gather

logger = logging.getLogger(__name__)


def all_gather_var_batchsize(tensor: Tensor) -> Tensor:
    world_size = dist.get_world_size()

    if world_size > 1:
        batch_sizes = [None] * world_size
        dist.all_gather_object(batch_sizes, tensor.shape[0])

        tensor_list = [
            torch.zeros((size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device) for size in batch_sizes
        ]
        dist.all_gather(tensor_list, tensor.contiguous())
        tensor_list[dist.get_rank()] = tensor
    else:
        tensor_list = [tensor]

    return tensor_list


def all_gather_var_batchsize_v2(tensor: Tensor, max_batch_size: int) -> Tensor:
    world_size = dist.get_world_size()

    if world_size > 1:
        # Pad tensor to max_batch_size rows
        pad_rows = max_batch_size - tensor.shape[0]
        pad = (0, 0) * (tensor.ndim - 1) + (0, pad_rows)  # pad last batch dim
        padded = torch.nn.functional.pad(tensor, pad)

        gathered = torch.zeros(world_size * max_batch_size, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
        dist.all_gather_into_tensor(gathered, padded.contiguous())

        # Build a local validity mask and all-gather it — still on GPU, no sync
        local_mask = torch.arange(max_batch_size, device=tensor.device) < tensor.shape[0]
        all_masks = torch.zeros(world_size * max_batch_size, dtype=torch.bool, device=tensor.device)
        dist.all_gather_into_tensor(all_masks, local_mask)

        # Slice on GPU using the gathered bool mask — no size scalars, no .tolist()
        tensor_list = [
            gathered[i * max_batch_size : (i + 1) * max_batch_size][
                all_masks[i * max_batch_size : (i + 1) * max_batch_size]
            ]
            for i in range(world_size)
        ]
        # Restore local rank's tensor with grad-carrying original
        tensor_list[dist.get_rank()] = tensor
    else:
        tensor_list = [tensor]

    return tensor_list


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
