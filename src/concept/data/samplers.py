import logging

import numpy as np
from lightning.fabric.utilities.distributed import DistributedSamplerWrapper as LightningDistributedSamplerWrapper
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class WithinGroupSampler(Sampler):
    def __init__(
        self,
        storage_idx,
        obs_list,
        batch_size,
        num_samples=None,
        shuffle=True,
        drop_last=True,
        stage="train",
        start_epoch=0,
    ):
        self.storage_idx = storage_idx
        self.obs_list = obs_list
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batches = None
        self.stage = stage
        self.current_epoch = start_epoch
        assert stage in ["train", "val", "test"], 'stage must be one of "train", "val", "test"'
        self._create_batches()

    def __len__(self):
        return sum([len(batch) for batch in self.batches])

    def __iter__(self):
        if self.stage == "train":
            self._create_batches()
        yield from np.hstack(self.batches)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def _validate_batches(self):
        storage_idx = self.storage_idx
        n_invalid_batches = sum([~(storage_idx[batch][0] == storage_idx[batch]).all() for batch in self.batches])
        assert n_invalid_batches == 0, f"Number of invalid batches: {n_invalid_batches}"

    def _create_batches(self):
        # Create RNG instance based on current epoch to ensure reproducibility
        if self.stage == "train":
            rng = np.random.default_rng(self.current_epoch)
            logger.info(f"Creating {self.stage} batches for epoch {self.current_epoch}...")
        else:
            rng = np.random.default_rng(42)  # for validation and test

        self.batches = []
        count = 0
        for obs in self.obs_list:
            n_obs = len(obs)
            for value in np.unique(obs):
                indices = np.argwhere(obs == value).flatten() + count
                if self.shuffle:
                    indices = rng.choice(indices, len(indices), replace=False)
                num_chunks = int(np.ceil(len(indices) / self.batch_size))
                batches = [indices[i * self.batch_size : (i + 1) * self.batch_size] for i in range(num_chunks)]
                # drop_last
                batches = batches[:-1] if len(batches[-1]) < self.batch_size else batches
                # shuffle(batches)
                self.batches.extend(batches)
            count += n_obs
        rng.shuffle(self.batches)
        if self.num_samples is not None:
            self.batches = self.batches[: self.num_samples // self.batch_size]
        self._validate_batches()


class DistributedSamplerWrapper(LightningDistributedSamplerWrapper):
    # Lightning calls this method when the epoch is set :
    # https://github.com/Lightning-AI/pytorch-lightning/blob/a967b6eba0556943ad1d7c2a8dc41f1da4f68b2d/pytorch_lightning/loops/fit_loop.py#L216

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self.dataset._sampler.set_epoch(epoch)
