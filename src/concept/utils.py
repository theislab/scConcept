import h5py
import filecmp
import logging
import os
import shutil
from datetime import timedelta
import torch
from lamin_dataloader import GeneIdTokenizer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def resolve_split_list(split_list: list, key: str | None = None) -> list:
    """
    Expand split config refs and flatten to a single list.

    If key is set (e.g. "train" or "val"), items that are dicts with that key
    are replaced by the key's list; otherwise items are kept as-is. Then any
    list-of-lists is flattened to a single list.
    """

    expanded = []
    for item in split_list:
        source_path = None
        if isinstance(item, dict):
            if "source_path" in item:
                source_path = item["source_path"]
            if key is not None and key in item:
                item = item[key]
        if isinstance(item, str):
            item = [item]
        if source_path:
            item = [os.path.join(source_path, file) for file in item]
        expanded.extend(item)

        split_list = expanded
    # if len(split_list) > 0 and isinstance(split_list[0], list):
    #     split_list = [item for sublist in split_list for item in sublist]
    return split_list


def load_pretrained_vocabulary(pretrained_vocabulary_dir: str, tokenizer: GeneIdTokenizer) -> list:
    import pandas as pd
    import glob

    # Load all CSV files from the directory
    csv_files = glob.glob(os.path.join(pretrained_vocabulary_dir, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {pretrained_vocabulary_dir}")

    logger.info(f"Loading {len(csv_files)} CSV files from {pretrained_vocabulary_dir}")

    # Merge all CSV files into a single dictionary
    pretrained_dict = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, index_col=0)
        for idx, row in df.iterrows():
            pretrained_dict[str(idx)] = row.values

    logger.info(f"Loaded {len(pretrained_dict)} gene embeddings from {len(csv_files)} files")

    pretrained_vocabulary = {}
    gene_names = tokenizer.decode(list(range(len(tokenizer.gene_mapping))))
    not_found_embeddings = []
    for idx, gene_name in enumerate(gene_names):
        if gene_name == tokenizer.CLS_VOCAB or gene_name == tokenizer.PAD_VOCAB:
            continue
        elif gene_name in pretrained_dict:
            pretrained_vocabulary[idx] = torch.FloatTensor(pretrained_dict[gene_name])
        else:
            not_found_embeddings.append(gene_name)
    if len(not_found_embeddings) > 0:
        logger.warning(f"Pretrained embeddings not found for {len(not_found_embeddings)} genes")
    else:
        logger.info(f"Pretrained embeddings found for all {len(gene_names)} genes")
    return pretrained_vocabulary


def get_start_epoch(cfg) -> int:
    if not cfg.initialize.resume:
        return 0

    # Try to get epoch from checkpoint file first
    ckpt_path = os.path.join(cfg.PATH.CHECKPOINT_ROOT, cfg.initialize.run_id, cfg.initialize.checkpoint)
    next_epoch = 1 if "epoch" in cfg.initialize.checkpoint else 0  # +1 because we want to start from the next epoch

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    start_epoch = int(checkpoint["epoch"]) + next_epoch
    logger.info(f"Resuming from epoch {start_epoch} (from checkpoint)")

    return start_epoch


def get_profiler(checkpoint_path: str):
    from lightning.pytorch.profilers import PyTorchProfiler
    from torch.profiler import schedule, tensorboard_trace_handler

    pl_profiler = PyTorchProfiler(
        dirpath=os.path.join(checkpoint_path, "profiler"),
        filename="profiler",
        # on_trace_ready=tensorboard_trace_handler(os.path.join(CHECKPOINT_PATH, 'profiler')), # Use this only for tensorboard
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        schedule=schedule(skip_first=170, wait=10, warmup=10, active=20, repeat=1),
        row_limit=-1,
        sort_by_key="cuda_time",
        # export_to_chrome=False
    )
    return pl_profiler


def copy_files(
    src_path: str, dst_path: str, filenames: list[str], compare_files: bool = False, force_copy: bool = False
):
    logger.info(f"Copying {len(filenames)} files from {src_path} to {dst_path} ...")
    if not os.path.exists(dst_path):
        os.makedirs(dst_path, exist_ok=True)
    copy_count = 0
    for file in filenames:
        src_file = os.path.join(src_path, file)
        dst_file = os.path.join(dst_path, file)
        if (
            not os.path.exists(dst_file)
            or (compare_files and not filecmp.cmp(src_file, dst_file, shallow=True))
            or force_copy
        ):
            shutil.copy(src_file, dst_file)
            copy_count += 1
    logger.info(f"{copy_count} new files copied successfully!")


def resume_wandb_config(bash_cfg: DictConfig) -> DictConfig:
    import wandb

    wandb.login()
    api = wandb.Api()
    run = api.run(f"{bash_cfg.wandb.entity}/{bash_cfg.wandb.project}/{bash_cfg.initialize.run_id}")
    logger.info(f"Resuming training for {run.id} ...")
    cfg = DictConfig(run.config)

    cfg = OmegaConf.merge(cfg, bash_cfg)

    if not cfg.initialize.create_new_run:
        os.environ["WANDB_TAGS"] = ",".join(run.tags) + "," + os.environ.get("WANDB_TAGS", "")

    # cfg.model.training.val_check_interval = float(cfg.model.training.val_check_interval + 0.1) # for a bug in pytorch-lightning
    cfg.model.training.val_check_interval = float(cfg.model.training.val_check_interval)
    cfg.model.training.limit_train_batches = float(cfg.model.training.limit_train_batches)
    return cfg


def _get_callbacks(checkpoint_path: str, max_steps: int):
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=os.path.join(checkpoint_path, "epochs"),
            filename="{epoch}",
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            save_top_k=-1,
            save_last="link",
        ),
        ModelCheckpoint(
            dirpath=os.path.join(checkpoint_path, "steps"),
            filename="{step}",
            every_n_train_steps=100_000 if max_steps > 100_000 else 10_000,
            save_top_k=-1,
            save_last="link",
        ),
        ModelCheckpoint(
            dirpath=os.path.join(checkpoint_path, "latest"),
            filename="{step}",
            every_n_train_steps=20_000,
            save_top_k=1,
            save_last="link",
        ),
    ]
    return callbacks
