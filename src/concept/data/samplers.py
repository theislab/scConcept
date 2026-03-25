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
        num_groups=None,
        sample_groups_equally=False,
        shuffle=True,
        drop_last=True,
        stage="train",
        start_epoch=0,
    ):
        self.sampling_key = sampling_key
        self.batch_size = batch_size
        self.num_samples = num_samples if num_samples is not None else len(sampling_key)
        self.num_groups = num_groups
        self.sample_groups_equally = sample_groups_equally
        self.shuffle = shuffle
        self.drop_last = drop_last
        assert drop_last, "The current implementation of WithinGroupSampler requires drop_last=True"
        self.batches = None
        self.stage = stage
        self.current_epoch = start_epoch
        self._batches_epoch = None
        assert stage in ["train", "val", "test"], 'stage must be one of "train", "val", "test"'
        self.seed = int(os.environ.get("PL_GLOBAL_SEED", 42))
        self._init_fixed_pool()
        self._create_batches()

    def __len__(self):
        return self.batches.size

    def __iter__(self):
        yield from self.batches.ravel()

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.stage == "train":
            self._create_batches()

    def _init_fixed_pool(self):
        """Determine a fixed pool of indices once at init, independent of epoch."""
        rng = np.random.default_rng(self.seed)

        # Drop NaN entries, then sort by group key
        nan_mask = pd.isnull(self.sampling_key)
        valid_pos = np.where(~nan_mask)[0]
        valid_keys = self.sampling_key[valid_pos]
        sort_order = np.argsort(valid_keys, kind="stable")
        sorted_pos = valid_pos[sort_order]
        sorted_keys = valid_keys[sort_order]

        # Group boundaries
        boundaries = np.where(sorted_keys[1:] != sorted_keys[:-1])[0] + 1
        group_slices = np.split(sorted_pos, boundaries)
        n_groups = len(group_slices)

        # Shuffle each group with the fixed RNG before any subsampling
        group_slices = [rng.permutation(g) for g in group_slices]

        if self.num_groups is not None and self.num_groups < n_groups:
            chosen = rng.choice(n_groups, size=self.num_groups, replace=False)
            group_slices = [group_slices[i] for i in chosen]
            n_groups = self.num_groups
            logger.info(f"Subsampled {n_groups} groups out of {len(boundaries) + 1} total groups")

        if self.sample_groups_equally:
            per_group = max(self.num_samples // n_groups, self.batch_size)
            group_slices = [g[:per_group] for g in group_slices]
        else:
            total = sum(len(g) for g in group_slices)
            if total > self.num_samples:
                group_slices = [g[: max(int(len(g) / total * self.num_samples), self.batch_size)] for g in group_slices]

        self._fixed_group_slices = [g.copy() for g in group_slices]

    def _validate_batches(self):
        if self.batches.size == 0:
            return
        # Single vectorized fancy-index: (num_batches, batch_size) keys
        batch_keys = self.sampling_key[self.batches]
        n_invalid = int(np.any(batch_keys != batch_keys[:, :1], axis=1).sum())
        assert n_invalid == 0, f"Number of invalid batches: {n_invalid}"

    def _create_batches(self):
        epoch_key = self.current_epoch if self.stage == "train" else "fixed"
        if self.batches is not None and self._batches_epoch == epoch_key:
            logger.info(f"Reusing cached {self.stage} batches for epoch {self.current_epoch}")
            return

        # Create RNG instance based on current epoch to ensure reproducibility
        if self.stage == "train":
            rng = np.random.default_rng(self.seed * 10_000 + self.current_epoch)
            logger.info(f"Creating train sampler for epoch {self.current_epoch}")
        else:
            rng = np.random.default_rng(self.seed)  # for validation and test

        all_batches = []
        for indices in self._fixed_group_slices:
            indices = indices.copy()  # preserve fixed pool across epochs
            if self.shuffle:
                rng.shuffle(indices)
            n_full = (len(indices) // self.batch_size) * self.batch_size
            if n_full == 0:
                continue
            all_batches.append(indices[:n_full].reshape(-1, self.batch_size))

        if all_batches:
            # Stack into a single (num_batches, batch_size) array
            self.batches = np.vstack(all_batches)
        else:
            self.batches = np.empty((0, self.batch_size), dtype=np.intp)

        if self.shuffle:
            # Shuffling rows of a 2D array is a single vectorized permutation
            rng.shuffle(self.batches)

        self._validate_batches()
        self._batches_epoch = self.current_epoch if self.stage == "train" else "fixed"
        if self.stage == "train":
            logger.info(f"Sampler created for training epoch {self.current_epoch}")


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

        indices = np.fromiter(self.sampler, dtype=np.intp, count=len(self.sampler))
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
