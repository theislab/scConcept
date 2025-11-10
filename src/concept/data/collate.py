import os
import numpy as np
import pandas as pd
from typing import Dict, List
from torch.utils.data import Dataset, default_collate, default_convert, get_worker_info
import torch.distributed as dist
from lamin_dataloader.dataset import BaseCollate
from abc import ABC, abstractmethod
import random
from pathlib import Path
import re


class Collate(BaseCollate):
    def __init__(self, 
                 tokenizer, 
                 panels_path,
                 max_tokens, 
                 min_tokens=0, 
                 split_input=True, 
                 variable_size=False, 
                 gene_sampling_strategy='top-nonzero',
                 panel_selection='random', 
                 panel_selection_mixed_prob=1.0, 
                 panel_filter_regex='.*', 
                 panel_size_min=None, 
                 panel_size_max=None,
                 panel_overlap=False,
                 anchor_panel_size=None, 
                 anchor_max_tokens=None, 
                 panel_max_drop_rate=None, 
                 feature_max_drop_rate=None,
                 model_speed_sanity_check=False,
                 ):
        super().__init__(PAD_TOKEN=tokenizer.PAD_TOKEN, 
                         max_tokens=max_tokens, 
                         gene_sampling_strategy=gene_sampling_strategy, 
        )
        
        self.tokenizer = tokenizer
        self.split_input = split_input
        self.panel_selection = panel_selection
        self.panel_selection_mixed_prob = panel_selection_mixed_prob
        if self.panel_selection != 'random':
            self.panel_dir = Path(panels_path)
            self.panel_names = [panel_name for panel_name in os.listdir(self.panel_dir) if re.search(panel_filter_regex, panel_name) and panel_name.endswith('.csv')]
            self.panels = [self.tokenizer.encode(pd.read_csv(self.panel_dir / panel_name)['Ensembl_ID'].values) 
                       for panel_name in self.panel_names]
            for i in range(len(self.panels)):
                print(f'Panel {self.panel_names[i]} size: {len(self.panels[i])} genes')
        self.panel_size_min = panel_size_min
        self.panel_size_max = panel_size_max
        self.panel_overlap = panel_overlap
        self.anchor_panel_size = anchor_panel_size
        self.anchor_max_tokens = anchor_max_tokens
        self.panel_max_drop_rate = panel_max_drop_rate
        self.gene_sampling_strategy = gene_sampling_strategy
        assert self.gene_sampling_strategy in ['random', 'top', 'random-nonzero', 'top-nonzero'], 'gene_sampling_strategy must be one of "random", "top", "random-nonzero", "top-nonzero"'
        self.feature_max_drop_rate = feature_max_drop_rate
        self.device_num = dist.get_rank() if dist.is_initialized() else 0
        self._rng = None
    
    # This is crucial when running multiple GPUs. 
    # It ensures that the random number generator is the same for each worker.
    @property
    def rng(self):
        if self._rng is None:
            if self.split_input:
                worker_info = get_worker_info()
                if worker_info: # In case of multi-process data loading
                    print(f'Device: {self.device_num}, Worker {worker_info.id} / {worker_info.num_workers}, seed: {worker_info.seed}')
                    self._rng = np.random.default_rng(seed=42 + worker_info.id)
                else:
                    self._rng = np.random.default_rng(42)    
            else:
                self._rng = np.random.default_rng(42)
        return self._rng


    def shared_feature_stats(self, batch):
        num_shared_featrues = []
        for i in range(len(batch)):
            for j in range(len(batch)):
                if i < j :
                    num_shared_featrues.append(np.intersect1d(batch[i]['tokens'], batch[j]['tokens']).size / np.union1d(batch[i]['tokens'], batch[j]['tokens']).size)
        
        print(f'Average % of shared features: %{np.median(num_shared_featrues)*100:.3f}')

    def log_int_samping(self, low, high):
        if low == high:
            return low
        randint = int(np.exp2(self.rng.uniform(np.log2(low), np.log2(high))))
        return max(min(randint, high), low)
    
    def int_samping(self, low, high):
        if low == high:
            return low
        randint = int(self.rng.uniform(low, high))
        return max(min(randint, high), low)
    
    
    def adapt_batch_size(self, batch,  batch_1, batch_2):
        max_length_1 = max(len(item['tokens']) for item in batch_1)
        max_length_2 = max(len(item['tokens']) for item in batch_2)
        
        if max_length_1 > 1000 or max_length_2 > 1000:
            new_batch_size = max(1, len(batch_1) // 4)
            batch = batch[:new_batch_size]
            batch_1 = batch_1[:new_batch_size]
            batch_2 = batch_2[:new_batch_size]
        
        return batch, batch_1, batch_2

    # def _custom_panel_selection(self, batch):
    #     tokens = batch[0]['tokens']
    #     count_nnz = batch[0]['count_nnz']
    #     if self.gene_sampling_strategy == 'random-nonzero':
    #         # perc_nnz = count_nnz / len(batch)
    #         # expressed_features = tokens[np.argsort(count_nnz)[::-1][:500]]
    #         available_panels = []
    #         panel_probs = []
    #         for i in range(len(self.panels)):
    #             panel = self.panels[i]
    #             available_panels.append(i)
    #             panel_probs.append(np.median(count_nnz[np.isin(tokens, panel)]))
    #             # panel_probs.append(np.intersect1d(self.panels[i], expressed_features).size / len(self.panels[i]))
    #             # if np.intersect1d(self.panels[i], expressed_features).size > (len(self.panels[i]) * 0.5):
    #             #     available_panels.append(i)    
    #             # available_panels.append((self.panel_names[i], len(self.panels[i]), f'{panel_probs[-1]:.2f}'))
            
    #         panel_probs = panel_probs // panel_probs.sum()
    #         # assert True==False, (batch[0]['dataset_name'], 'len(tokens)', len(tokens), 'panels', list(zip(self.panel_names, panel_probs)), count_nnz)
    #         return available_panels, panel_probs

    def _get_predesigned_panel(self, batch):

        i = self.rng.integers(0, len(self.panels))
        # available_panels, panel_probs = self._custom_panel_selection(batch)
        # i = self.rng.choice(available_panels, p=panel_probs)
        panel = self.panels[i]
        if self.panel_max_drop_rate is not None and self.panel_max_drop_rate > 0:
            panel_max_drop_rate = self.rng.uniform(0, self.panel_max_drop_rate)
            drop_mask = self.rng.uniform(size=len(panel)) > panel_max_drop_rate
            panel = panel[drop_mask]
        return panel, self.panel_names[i]
    
    def __call__(self, batch):
        n_tokens = len(batch[0]['tokens'])
        permute = self.rng.permutation(n_tokens)
        batch_permute = [{'tokens': item['tokens'][permute], 
                          'values': item['values'][permute], 
                        #   'count_nnz': item['count_nnz'][permute],
                        #   'dataset_name': item['dataset_name'],
                          } for item in batch]
        
        if self.split_input:
            
            n_tokens = len(batch_permute[0]['tokens'])
            panel_indices = np.arange(n_tokens)

            panel_name = 'random'
            panel_overlap = self.rng.uniform() <= float(self.panel_overlap)
            
            if self.panel_selection == 'random' or (self.panel_selection == 'mixed' and self.rng.uniform() <= self.panel_selection_mixed_prob) or n_tokens < 10_000:
                n_tokens_available = n_tokens if panel_overlap else max((n_tokens - self.panel_size_min), 0)
                panel_size_1 = self.log_int_samping(min(self.panel_size_min, n_tokens_available), min(self.panel_size_max, n_tokens_available))
                panel_idx_1 = self.rng.choice(panel_indices, panel_size_1, replace=False)
                # print(f'Panel_1 random size: {len(panel_idx_1)}')
            else:
                panel, panel_name = self._get_predesigned_panel(batch_permute)
                panel_idx_1 = np.where(np.isin(batch_permute[0]['tokens'], panel))[0]
                panel_size_1 = len(panel_idx_1)
                # print(f'Panel_1 {self.panel_names[i]} predefined size: {len(panel_idx_1)}')
            
            # Use anchor for training
            # if self.anchor_panel_size is None:
            if panel_overlap:
                panel_size_2 = self.log_int_samping(min(self.panel_size_min, n_tokens), min(self.panel_size_max, n_tokens))
                panel_idx_2 = self.rng.choice(panel_indices, panel_size_2, replace=False)
                # print(f'Panel_2 random size: {len(panel_idx_2)}, shared: {np.intersect1d(panel_idx_1, panel_idx_2).size}')
            else:
                panel_size_2 = self.log_int_samping(min(self.panel_size_min, n_tokens - panel_size_1), min(self.panel_size_max, n_tokens - panel_size_1))
                panel_idx_2 = self.rng.choice(np.setdiff1d(panel_indices, panel_idx_1, assume_unique=True), panel_size_2, replace=False)
                assert np.intersect1d(panel_idx_1, panel_idx_2).size == 0, 'Panels overlap'
            # else:
            #     panel_size_2 =  min(self.anchor_panel_size, n_tokens - panel_size_1)
            #     panel_idx_2 = self.rng.choice(np.setdiff1d(panel_indices, panel_idx_1, assume_unique=True), panel_size_2, replace=False)
            #     assert np.intersect1d(panel_idx_1, panel_idx_2).size == 0, 'Panels overlap'
            
                        
            batch_1 = [{'tokens': item['tokens'][panel_idx_1], 'values': item['values'][panel_idx_1]} for item in batch_permute]
            batch_2 = [{'tokens': item['tokens'][panel_idx_2], 'values': item['values'][panel_idx_2]} for item in batch_permute]

            # The following can be optimized by only passing one panel per batch
            panel_1 = [item['tokens'] for item in batch_1]
            panel_2 = [item['tokens'] for item in batch_2]

            batch_1 = [self.select_features(item, self.feature_max_drop_rate) for item in batch_1]
            batch_2 = [self.select_features(item, self.feature_max_drop_rate) for item in batch_2]

            max_lenght_1 = max([len(item['tokens']) for item in batch_1])
            max_lenght_1 = min(max_lenght_1, self.max_tokens - 1) # todo
            # min_lenght_1 = min([len(item['tokens']) for item in batch_1])
            # max_lenght_1 = min(self.int_samping(min_lenght_1, max_lenght_1), self.max_tokens)
            max_lenght_2 = max([len(item['tokens']) for item in batch_2])
            max_lenght_2 = min(max_lenght_2, self.max_tokens - 1) # todo
            
            # Use anchor for training
            # if self.anchor_panel_size is not None:
            #     max_lenght_2 = min([len(item['tokens']) for item in batch_2])
            #     max_lenght_2 = min(max_lenght_2, self.anchor_max_tokens)
            
            batch_1 = [self.resize_and_pad(item, max_lenght_1) for item in batch_1]
            batch_2 = [self.resize_and_pad(item, max_lenght_2) for item in batch_2]
            
            # batch, batch_1, batch_2 = self.adapt_batch_size(batch, batch_1, batch_2)
            # self.shared_feature_stats(batch_1)
            
            
            tokens_1 = [item['tokens'].astype(np.int64) for item in batch_1]
            values_1 = [item['values'].astype(np.float32) for item in batch_1]
            tokens_2 = [item['tokens'].astype(np.int64) for item in batch_2]
            values_2 = [item['values'].astype(np.float32) for item in batch_2]
            
            return {'tokens_1': default_collate(tokens_1),
                    'values_1': default_collate(values_1),
                    'tokens_2': default_collate(tokens_2),
                    'values_2': default_collate(values_2),
                    'panel_1' : default_collate(panel_1),
                    'panel_2' : default_collate(panel_2),
                    'panel_name': panel_name,
                    **{key: default_collate([item[key] for item in batch]) for key in batch[0].keys() if key not in ['tokens', 'values']}
            }
            # return {'tokens_1': default_convert(tokens_1),
            #         'values_1': default_convert(values_1),
            #         'tokens_2': default_convert(tokens_2),
            #         'values_2': default_convert(values_2),
            #         'panel_1' : default_collate(panel_1),
            #         'panel_2' : default_collate(panel_2),
            #         **{key: default_collate([item[key] for item in batch]) for key in batch[0].keys() if key not in ['tokens', 'values']}
            # }
        else:
            batch_ = [self.select_features(item) for item in batch_permute]
            
            max_lenght = max([len(item['tokens']) for item in batch_])
            
            max_lenght = min(max_lenght, self.max_tokens - 1)

            batch_ = [self.resize_and_pad(item, max_lenght) for item in batch_]
            
            tokens, values = [item['tokens'].astype(np.int64) for item in batch_], [item['values'].astype(np.float32) for item in batch_]
        
            return {'tokens': default_collate(tokens),
                    'values': default_collate(values),
                    **{key: default_collate([item[key] for item in batch]) for key in batch[0].keys() if key not in ['tokens', 'values']}
            }


