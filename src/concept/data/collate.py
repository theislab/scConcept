import logging
import os
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from lamin_dataloader import BaseCollate
from torch.utils.data import get_worker_info

logger = logging.getLogger(__name__)


class Collate(BaseCollate):
    def __init__(
        self,
        tokenizer,
        max_tokens,
        stage="train",
        min_tokens=0,
        panels_path=None,
        split_input=False,
        gene_sampling_strategy="top-nonzero",
        panel_selection="random",
        panel_selection_mixed_prob=1.0,
        panel_filter_regex=".*",
        panel_size_min=None,
        panel_size_max=None,
        panel_overlap=False,
        panel_max_drop_rate=None,
        feature_max_drop_rate=None,
        qc_threshold=None,
        max_total_seq_length=float("inf"),
        model_speed_sanity_check=False,
    ):
        super().__init__(
            PAD_TOKEN=tokenizer.PAD_TOKEN,
            max_tokens=max_tokens,
            gene_sampling_strategy=gene_sampling_strategy,
        )

        self.tokenizer = tokenizer
        self.stage = stage
        assert self.stage in ["train", "val", "test"], (
            f"Invalid stage: {self.stage}, must be one of 'train', 'val', 'test'"
        )
        self.split_input = split_input
        self.panel_selection = panel_selection
        self.panel_selection_mixed_prob = panel_selection_mixed_prob
        if self.panel_selection != "random":
            self.load_panels(panels_path, panel_filter_regex, stage)
        self.panel_size_min = panel_size_min
        self.panel_size_max = panel_size_max
        self.panel_overlap = panel_overlap
        self.panel_max_drop_rate = panel_max_drop_rate
        self.gene_sampling_strategy = gene_sampling_strategy
        assert self.gene_sampling_strategy in ["random", "top", "random-nonzero", "top-nonzero"], (
            'gene_sampling_strategy must be one of "random", "top", "random-nonzero", "top-nonzero"'
        )
        self.feature_max_drop_rate = feature_max_drop_rate
        self.qc_threshold = qc_threshold
        self.max_total_seq_length = max_total_seq_length
        self.device_num = dist.get_rank() if dist.is_initialized() else 0
        self._rng = None

    # This is crucial when running multiple GPUs.
    # It ensures that the random number generator is the same for each worker across GPU processes.
    @property
    def rng(self):
        if self._rng is None:
            seed = int(os.environ.get("PL_GLOBAL_SEED", 42))
            if self.split_input:
                worker_info = get_worker_info()
                if worker_info:  # In case of multi-process data loading
                    if worker_info.id == 0:
                        logger.info(
                            f"Device: {self.device_num}, Worker {worker_info.id} / {worker_info.num_workers}, seed: {worker_info.seed}"
                        )
                    seed = worker_info.seed
            self._rng = np.random.default_rng(seed)
        return self._rng

    def load_panels(self, panels_path, panel_filter_regex, stage="train"):
        panels_path = Path(panels_path)
        # Load panels from organisms-specific subdirectories
        self.panels_dict = {}
        self.panel_names_dict = {}

        # Load panels from each organisms subdirectory
        subdirs = [d for d in panels_path.iterdir() if d.is_dir()]

        for organism_dir in subdirs:
            organism_name = organism_dir.name
            panel_files = [
                panel_file
                for panel_file in os.listdir(organism_dir)
                if re.search(panel_filter_regex, panel_file) and panel_file.endswith(".csv")
            ]

            if panel_files and organism_name in self.tokenizer.species:
                panels = [
                    self.tokenizer.encode(pd.read_csv(organism_dir / panel_file)["Ensembl_ID"].values, organism_name)
                    for panel_file in panel_files
                ]
                self.panels_dict[organism_name] = panels
                self.panel_names_dict[organism_name] = panel_files

                if stage == "train":
                    panel_sizes = [len(panel) for panel in panels]
                    logger.info(
                        f"Organism {organism_name}: loaded {len(panel_files)} panels "
                        f"(genes per panel - min: {min(panel_sizes)}, "
                        f"median: {int(np.median(panel_sizes))}, max: {max(panel_sizes)})"
                    )

    def shared_feature_stats(self, batch):
        num_shared_featrues = []
        for i in range(len(batch)):
            for j in range(len(batch)):
                if i < j:
                    num_shared_featrues.append(
                        np.intersect1d(batch[i]["tokens"], batch[j]["tokens"]).size
                        / np.union1d(batch[i]["tokens"], batch[j]["tokens"]).size
                    )

        logger.info(f"Average % of shared features: {np.median(num_shared_featrues) * 100:.3f}")

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

    def adapt_batch_size(self, seq_length_1, seq_length_2):
        seq_length = np.array(seq_length_1) + np.array(seq_length_2)
        seq_length_cumsum = np.cumsum(seq_length)

        new_batch_size = len(seq_length_1)
        if seq_length_cumsum[-1] > self.max_total_seq_length:
            new_batch_size = np.searchsorted(seq_length_cumsum, self.max_total_seq_length)
            return new_batch_size

        return new_batch_size

    def qc_mask(self, seq_length_1, seq_length_2, panel_size_1, panel_size_2):
        seq_length_1, seq_length_2 = np.array(seq_length_1), np.array(seq_length_2)
        mask_1 = seq_length_1 > panel_size_1 * self.qc_threshold
        mask_2 = seq_length_2 > panel_size_2 * self.qc_threshold
        mask = mask_1 & mask_2
        return mask

    def _apply_items_mask(self, items_mask, batch_1, batch_2, seq_length_1, seq_length_2):
        for i in range(len(batch_1)):
            if not items_mask[i]:
                batch_1[i] = {"tokens": np.array([]), "values": np.array([])}
                batch_2[i] = {"tokens": np.array([]), "values": np.array([])}
                seq_length_1[i] = 0
                seq_length_2[i] = 0

    def _get_predesigned_panel(self, batch, species):
        # Randomly select a panel from that species
        panels = self.panels_dict[species]
        panel_names = self.panel_names_dict[species]
        i = self.rng.integers(0, len(panels))

        panel = panels[i]
        if self.panel_max_drop_rate is not None and self.panel_max_drop_rate > 0:
            panel_max_drop_rate = self.rng.uniform(0, self.panel_max_drop_rate)
            drop_mask = self.rng.uniform(size=len(panel)) > panel_max_drop_rate
            panel = panel[drop_mask]
        return panel, f"{species}/{panel_names[i]}"

    def __call__(self, batch):
        n_tokens = len(batch[0]["tokens"])
        permute = self.rng.permutation(n_tokens)
        batch_permute = [
            {
                "tokens": item["tokens"][permute],
                "values": item["values"][permute],
            }
            for item in batch
        ]

        if self.split_input:
            species = [item.get("species", None) for item in batch]
            assert species is not None, "species is None"
            assert len(set(species)) == 1, "Multiple species in the same batch is not supported"
            species = species[0]

            n_tokens = len(batch_permute[0]["tokens"])
            panel_indices = np.arange(n_tokens)
            is_targetted_assay = n_tokens < 10_000

            panel_name_1, panel_name_2 = "random", "random"
            panel_overlap = self.rng.uniform() <= float(self.panel_overlap)

            if (
                self.panel_selection == "random"
                or (self.panel_selection == "mixed" and self.rng.uniform() <= self.panel_selection_mixed_prob)
                or is_targetted_assay
                or species not in self.panels_dict
            ):
                n_tokens_available = n_tokens if panel_overlap else max((n_tokens - self.panel_size_min), 0)
                panel_size_1 = self.log_int_samping(
                    min(self.panel_size_min, n_tokens_available), min(self.panel_size_max, n_tokens_available)
                )
                panel_idx_1 = self.rng.choice(panel_indices, panel_size_1, replace=False)
            else:
                panel, panel_name_1 = self._get_predesigned_panel(batch_permute, species)
                panel_idx_1 = np.where(np.isin(batch_permute[0]["tokens"], panel))[0]
                panel_size_1 = len(panel_idx_1)

            if panel_overlap:
                panel_size_2 = self.log_int_samping(
                    min(self.panel_size_min, n_tokens), min(self.panel_size_max, n_tokens)
                )
                panel_idx_2 = self.rng.choice(panel_indices, panel_size_2, replace=False)
            else:
                panel_size_2 = self.log_int_samping(
                    min(self.panel_size_min, n_tokens - panel_size_1), min(self.panel_size_max, n_tokens - panel_size_1)
                )
                panel_idx_2 = self.rng.choice(
                    np.setdiff1d(panel_indices, panel_idx_1, assume_unique=True), panel_size_2, replace=False
                )
                assert np.intersect1d(panel_idx_1, panel_idx_2).size == 0, "Panels overlap"

            batch_1 = [
                {"tokens": item["tokens"][panel_idx_1], "values": item["values"][panel_idx_1]} for item in batch_permute
            ]
            batch_2 = [
                {"tokens": item["tokens"][panel_idx_2], "values": item["values"][panel_idx_2]} for item in batch_permute
            ]

            panel_1 = batch_1[0]["tokens"]
            panel_2 = batch_2[0]["tokens"]

            batch_1 = [self.select_features(item, self.feature_max_drop_rate) for item in batch_1]
            batch_2 = [self.select_features(item, self.feature_max_drop_rate) for item in batch_2]

            seq_length_1 = [min(len(item["tokens"]), self.max_tokens) for item in batch_1]
            seq_length_2 = [min(len(item["tokens"]), self.max_tokens) for item in batch_2]

            items_mask = np.ones(len(batch_1), dtype=bool)

            if self.stage == "train" and self.qc_threshold is not None:
                qc_mask = self.qc_mask(seq_length_1, seq_length_2, len(panel_1), len(panel_2))
                items_mask &= qc_mask
                self._apply_items_mask(items_mask, batch_1, batch_2, seq_length_1, seq_length_2)

            if (
                self.stage == "train"
                and self.max_total_seq_length is not None
                and self.max_total_seq_length < float("inf")
            ):
                batch_size = self.adapt_batch_size(seq_length_1, seq_length_2)
                if batch_size < len(batch_1):
                    items_mask[batch_size:] = False
                    self._apply_items_mask(items_mask, batch_1, batch_2, seq_length_1, seq_length_2)

            max_lenght_1 = max(seq_length_1)
            max_lenght_2 = max(seq_length_2)
            batch_1 = [self.resize_and_pad(item, max_lenght_1) for item in batch_1]
            batch_2 = [self.resize_and_pad(item, max_lenght_2) for item in batch_2]

            # self.shared_feature_stats(batch_1)

            tokens_1 = [item["tokens"].astype(np.int64) for item in batch_1]
            values_1 = [item["values"].astype(np.float32) for item in batch_1]
            tokens_2 = [item["tokens"].astype(np.int64) for item in batch_2]
            values_2 = [item["values"].astype(np.float32) for item in batch_2]

            return {
                "tokens_1": torch.from_numpy(np.stack(tokens_1)),
                "values_1": torch.from_numpy(np.stack(values_1)),
                "tokens_2": torch.from_numpy(np.stack(tokens_2)),
                "values_2": torch.from_numpy(np.stack(values_2)),
                "panel_1": torch.from_numpy(panel_1),
                "panel_2": torch.from_numpy(panel_2),
                "panel_name_1": panel_name_1,
                "panel_name_2": panel_name_2,
                "seq_length_1": np.array(seq_length_1),
                "seq_length_2": np.array(seq_length_2),
                "items_mask": torch.from_numpy(items_mask),
                **{
                    key: (
                        np.array([item[key] for item in batch])
                        if isinstance(batch[0][key], str)
                        else torch.from_numpy(np.stack([item[key] for item in batch]))
                    )
                    for key in batch[0].keys()
                    if key not in ["tokens", "values"]
                },
            }

        else:
            batch_ = [self.select_features(item) for item in batch_permute]

            max_lenght = max([len(item["tokens"]) for item in batch_])

            max_lenght = min(max_lenght, self.max_tokens)

            batch_ = [self.resize_and_pad(item, max_lenght) for item in batch_]

            tokens, values = (
                [item["tokens"].astype(np.int64) for item in batch_],
                [item["values"].astype(np.float32) for item in batch_],
            )

            return {
                "tokens": torch.from_numpy(np.stack(tokens)),
                "values": torch.from_numpy(np.stack(values)),
                **{
                    key: (
                        np.array([item[key] for item in batch])
                        if isinstance(batch[0][key], str)
                        else torch.from_numpy(np.stack([item[key] for item in batch]))
                    )
                    for key in batch[0].keys()
                    if key not in ["tokens", "values"]
                },
            }
