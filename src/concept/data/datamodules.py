import os
from typing import Dict, List


import lightning as L
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from lamin_dataloader.dataset import TokenizedDataset, Tokenizer
from concept.data.collate import Collate
from concept.data.samplers import WithinGroupSampler
from lamin_dataloader.collections import InMemoryCollection
from lightning.fabric.utilities.distributed import DistributedSamplerWrapper
import multiprocessing
from anndata import AnnData

class AnnDataModule(L.LightningDataModule):
    def __init__(
        self,
        split: Dict,
        panels_path: str,
        tokenizer: Tokenizer,
        columns: List[str],
        precomp_embs_key: str = None,
        normalization: str = 'log1p',
        gene_sampling_strategy: str = 'top-nonzero',
        model_speed_sanity_check: bool = False,
        dataset_kwargs: Dict = {},
        dataloader_kwargs: Dict = {},
        val_loader_names = []
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.panels_path = panels_path
        self.gene_sampling_strategy = gene_sampling_strategy
        self.model_speed_sanity_check = model_speed_sanity_check
        self.val_loader_names = val_loader_names
        self.dataloader_kwargs = dataloader_kwargs

        dataset_kwargs_shared = {'obs_keys': columns, 
                                 'obsm_key': precomp_embs_key,
                                 'tokenizer': tokenizer, 
                                 'normalization': normalization
                                 }
                

        if 'train' in split and split['train'] is not None and 'train' in dataset_kwargs:
            within_group_sampling = dataloader_kwargs['train']['within_group_sampling']
            keys_to_cache = [within_group_sampling] if within_group_sampling else []
            self.train_collate_fn = self._get_collate_fn(dataset_kwargs['train'], split_input=True)
            
            if isinstance(split['train'][0], AnnData):
                assert within_group_sampling == 'dataset', 'within_group_sampling must be dataset for AnnData objects'
                # Use InMemoryCollection for AnnData objects
                collection = InMemoryCollection(
                    adata_list=split['train'],
                    obs_keys=columns,
                    layers_keys=['X'],
                    obsm_keys=precomp_embs_key,
                    keys_to_cache=keys_to_cache
                )
            else:
                # Use LaminDiskCollection for file paths
                from lamin_dataloader.lamin_disk_collection import LaminDiskCollection
                join = None if within_group_sampling else "outer"
                collection = LaminDiskCollection(split['train'], layers_keys="X", obs_keys=columns, keys_to_cache=keys_to_cache, join=join, encode_labels=True, parallel=True, obsm_keys=precomp_embs_key)
            
            self.train_dataset = TokenizedDataset(**{'collection': collection, **dataset_kwargs_shared, **dataset_kwargs['train']})
        if 'val' in split and split['val'] is not None and 'val' in dataset_kwargs:
            self.val_datasets = {}
            for val_name, val_kwargs in dataset_kwargs['val'].items():
                within_group_sampling = dataloader_kwargs['val'][val_name]['within_group_sampling']
                keys_to_cache = [within_group_sampling] if within_group_sampling else []
                val_collate_fn = self._get_collate_fn(val_kwargs, split_input=True)
                
                if isinstance(split['val'][0], AnnData):
                    assert within_group_sampling == 'dataset', 'within_group_sampling must be dataset for AnnData objects'
                    # Use InMemoryCollection for AnnData objects
                    collection = InMemoryCollection(
                        adata_list=split['val'],
                        obs_keys=columns,
                        layers_keys=['X'],
                        obsm_keys=precomp_embs_key,
                        keys_to_cache=keys_to_cache
                    )
                else:
                    # Use LaminDiskCollection for file paths
                    from lamin_dataloader.lamin_disk_collection import LaminDiskCollection
                    join = None if within_group_sampling else "outer"
                    collection = LaminDiskCollection(split['val'], layers_keys="X", obs_keys=columns, keys_to_cache=keys_to_cache, join=join, encode_labels=True, parallel=True, obsm_keys=precomp_embs_key)
                
                dataset = TokenizedDataset(**{'collection': collection, **dataset_kwargs_shared, **val_kwargs})
                self.val_datasets[val_name] = (dataset, val_collate_fn)
        if 'test' in split and split['test'] is not None and 'test' in dataset_kwargs:
            keys_to_cache = None
            self.test_collate_fn = self._get_collate_fn(dataset_kwargs['test'], split_input=False)
            
            if isinstance(split['test'][0], AnnData):
                # Use InMemoryCollection for AnnData objects
                assert within_group_sampling == 'dataset', 'within_group_sampling must be dataset for AnnData objects'
                collection = InMemoryCollection(
                    adata_list=split['test'],
                    obs_keys=columns,
                    layers_keys=['X'],
                    obsm_keys=precomp_embs_key,
                    keys_to_cache=keys_to_cache
                )
            else:
                # Use LaminDiskCollection for file paths
                from lamin_dataloader.lamin_disk_collection import LaminDiskCollection
                collection = LaminDiskCollection(split['test'], layers_keys="X", obs_keys=columns, keys_to_cache=keys_to_cache, join=None, encode_labels=True, parallel=True)
            
            self.test_dataset = TokenizedDataset(**{'collection': collection, **dataset_kwargs_shared, **dataset_kwargs['test']})

        self._val_dataloader = None
    
    def _get_collate_fn(self, dataset_kwargs, split_input):
        keys_to_pop = [
            'max_tokens', 'min_tokens', 'variable_size', 'panel_selection', 'panel_selection_mixed_prob',
            'panel_filter_regex', 'panel_size_min', 'panel_size_max', 'panel_overlap',
            'anchor_panel_size', 'anchor_max_tokens', 'panel_max_drop_rate', 'feature_max_drop_rate',
        ]

        collate_kwargs = {
            'tokenizer': self.tokenizer,
            'panels_path': self.panels_path,
            'split_input': split_input,
            'gene_sampling_strategy': self.gene_sampling_strategy,
            'model_speed_sanity_check': self.model_speed_sanity_check,
            **{key: dataset_kwargs.pop(key) for key in keys_to_pop if key in dataset_kwargs}
        }
        return Collate(**collate_kwargs)
    
    def _get_dataloader(self, dataset, dataloader_kwargs, collate_fn, stage):
        sampling_key = dataloader_kwargs.pop('within_group_sampling')
        num_replicas = dist.get_world_size() if torch.distributed.is_initialized() else 1
        batch_size = dataloader_kwargs.pop('batch_size') // num_replicas
        shuffle = dataloader_kwargs.pop('shuffle')
        drop_last = dataloader_kwargs.pop('drop_last')
        num_samples = dataloader_kwargs.pop('num_samples')
        num_workers = dataloader_kwargs.pop('num_workers')
        num_workers = min(int(os.getenv("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count())), num_workers)
        
        assert drop_last == True, 'drop_last must be True during training and validation'
        assert shuffle == True, 'shuffle must be True during training and validation'
        
        if num_samples is not None and num_samples >= len(dataset):
            print(f'Warning: num_samples ({num_samples}) is greater than or equal to the number of samples in the dataset ({len(dataset)}).')

        if sampling_key:
            sampler = WithinGroupSampler(dataset.collection.storage_idx, dataset.collection._cached_obs[sampling_key], batch_size * num_replicas, num_samples, shuffle=shuffle, drop_last=drop_last, stage=stage)
        else:
            sampler = RandomSampler(dataset, num_samples=num_samples)
        
        if torch.distributed.is_initialized():
            sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=False)

        # torch_worker_init_fn may not exist for InMemoryCollection
        worker_init_fn = getattr(dataset.collection, 'torch_worker_init_fn', None)
        
        dataloader = DataLoader(dataset, 
                                sampler=sampler, 
                                batch_size=batch_size,
                                drop_last=drop_last,
                                worker_init_fn=worker_init_fn,
                                collate_fn=collate_fn,
                                num_workers=num_workers,
                                persistent_workers=(num_workers > 0),
                                **dataloader_kwargs)
        print(f'Creating {stage} dataloader by {len(dataloader)} batches of size {batch_size*num_replicas} taking {len(dataloader)*batch_size*num_replicas} samples from {len(dataset)} total samples; num_replicas={num_replicas}; sum of indices: {sum(dataset.collection.indices)}; num_workers={num_workers}')
        return dataloader
        
    def train_dataloader(self):
        dataloader_kwargs = self.dataloader_kwargs['train']
        dataloader = self._get_dataloader(self.train_dataset, dataloader_kwargs, self.train_collate_fn, 'train')
        return dataloader

    def val_dataloader(self):
        if self._val_dataloader is not None:
            return self._val_dataloader
        
        self._val_dataloader = []
        for val_name in self.val_loader_names:
            val_dataset, val_collate_fn = self.val_datasets[val_name]
            dataloader_kwargs = self.dataloader_kwargs['val'][val_name]
            dataloader = self._get_dataloader(val_dataset, dataloader_kwargs, val_collate_fn, 'val')
            self._val_dataloader.append(dataloader)
        return self._val_dataloader
        
    def test_dataloader(self):
        dataloader_kwargs = self.dataloader_kwargs['test']
        
        assert dataloader_kwargs['shuffle'] == False, 'shuffle should be false for test dataloader'
        assert dataloader_kwargs['drop_last'] == False, 'drop_last should be false for test dataloader'
        # torch_worker_init_fn may not exist for InMemoryCollection
        worker_init_fn = getattr(self.test_dataset.collection, 'torch_worker_init_fn', None)
        dataloader = DataLoader(self.test_dataset, 
                                worker_init_fn=worker_init_fn, 
                                collate_fn=self.test_collate_fn, 
                                **dataloader_kwargs)
        print(f'Creating test dataloader by {len(dataloader)} batches of size {dataloader_kwargs["batch_size"]} over {len(self.test_dataset)} samples; sum of indices: {sum(self.test_dataset.collection.indices)}')
        return dataloader