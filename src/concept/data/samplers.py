import logging
import math
import os
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch.distributed as dist
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class WithinGroupSampler(Sampler):
    def __init__(
        self,
        sampling_key,
        batch_size,
        num_samples=None,
        sample_groups_equally=False,
        shuffle=True,
        drop_last=True,
        stage="train",
        start_epoch=0,
    ):
        self.sampling_key = sampling_key
        self.batch_size = batch_size
        self.num_samples = num_samples if num_samples is not None else len(sampling_key)
        self.sample_groups_equally = sample_groups_equally
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batches = None
        self.stage = stage
        self.current_epoch = start_epoch
        assert stage in ["train", "val", "test"], 'stage must be one of "train", "val", "test"'
        self.seed = int(os.environ.get("PL_GLOBAL_SEED", 42))
        self._create_batches()

    def __len__(self):
        return sum([len(batch) for batch in self.batches])

    def __iter__(self):
        yield from np.hstack(self.batches)
        if self.stage == "train":
            self._create_batches()

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def _validate_batches(self):
        n_invalid_batches = sum([~(self.sampling_key[batch][0] == self.sampling_key[batch]).all() for batch in self.batches])
        assert n_invalid_batches == 0, f"Number of invalid batches: {n_invalid_batches}"

    def _create_batches(self):
        # Create RNG instance based on current epoch to ensure reproducibility
        if self.stage == "train":
            rng = np.random.default_rng(self.seed * 10_000 + self.current_epoch)
        else:
            rng = np.random.default_rng(self.seed)  # for validation and test

        self.batches = []
        unique_values = np.unique(self.sampling_key)
        unique_values = pd.Series(unique_values).dropna().values
        for value in unique_values:
            indices = np.argwhere(self.sampling_key == value).flatten()
            if self.shuffle:
                indices = rng.choice(indices, len(indices), replace=False)
            if self.sample_groups_equally:
                indices = indices[:max(self.num_samples // len(unique_values), self.batch_size)]
            num_chunks = int(np.ceil(len(indices) / self.batch_size))
            batches = [indices[i * self.batch_size : (i + 1) * self.batch_size] for i in range(num_chunks)]
            if self.drop_last:
                batches = batches[:-1] if len(batches[-1]) < self.batch_size else batches
            self.batches.extend(batches)
        if self.shuffle:
            rng.shuffle(self.batches)
        if self.num_samples is not None:
            self.batches = self.batches[: self.num_samples // self.batch_size]
        self._validate_batches()


class DistributedSamplerWrapper(Sampler):
    def __init__(
        self,
        sampler: Sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")

        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

    def __iter__(self) -> Iterator:
        # Get all indices from the wrapped sampler

        num_samples = self.__len__()

        total_size = num_samples * self.num_replicas

        indices = list(self.sampler)
        # remove tail of data to make it evenly divisible.
        indices = indices[:total_size]

        assert len(indices) == total_size, f"Expected {total_size} indices, got {len(indices)}"

        # subsample
        indices = indices[self.rank : total_size : self.num_replicas]
        assert len(indices) == num_samples, f"Expected {num_samples} indices, got {len(indices)}"

        return iter(indices)

    def __len__(self) -> int:
        # Compute length dynamically based on current sampler state
        total_samples = len(self.sampler)
        if total_samples % self.num_replicas != 0:
            num_samples = math.ceil((total_samples - self.num_replicas) / self.num_replicas)
        else:
            num_samples = math.ceil(total_samples / self.num_replicas)

        return num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        This method forwards the epoch information to the wrapped sampler
        if it has a `set_epoch` method, allowing the wrapped sampler to
        update its internal state for proper randomization across epochs.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        # Pass epoch to wrapped sampler if it supports it
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
