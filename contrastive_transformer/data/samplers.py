
from numpy.random import choice
from random import shuffle
from torch.utils.data import Sampler
import numpy as np

np.random.seed(42)

        
class WithinGroupSampler(Sampler):
    def __init__(self, storage_idx, obs_list, batch_size, num_samples=None, shuffle=True, drop_last=True, stage='train'):
        self.storage_idx = storage_idx
        self.obs_list = obs_list
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batches = None
        self._create_batches()
        self.stage = stage
        assert stage in ['train', 'val', 'test'], 'stage must be one of "train", "val", "test"'
    
    def __len__(self):
        return sum([len(batch) for batch in self.batches])

    def __iter__(self):
        if self.stage == 'train':
            self._create_batches()
        yield from np.hstack(self.batches)

    def _validate_batches(self):
        storage_idx = self.storage_idx
        n_invalid_batches = sum([~(storage_idx[batch][0]==storage_idx[batch]).all() for batch in self.batches])
        assert n_invalid_batches == 0, f'Number of invalid batches: {n_invalid_batches}'        

    def _create_batches(self):
        self.batches = []
        count = 0
        for obs in self.obs_list:
            n_obs = len(obs)
            for value in np.unique(obs):
                indices = np.argwhere(obs == value).flatten() + count
                if self.shuffle:
                    indices = choice(indices, len(indices), replace=False)
                num_chunks = int(np.ceil(len(indices) / self.batch_size))
                batches = [indices[i * self.batch_size: (i + 1) * self.batch_size] for i in range(num_chunks)]
                # drop_last
                batches = batches[:-1] if len(batches[-1]) < self.batch_size else batches
                # shuffle(batches)
                self.batches.extend(batches)
            count += n_obs
        shuffle(self.batches)
        if self.num_samples is not None:
            self.batches = self.batches[:self.num_samples//self.batch_size]
        self._validate_batches()