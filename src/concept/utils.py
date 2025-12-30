import filecmp
import logging
import os
import shutil
from datetime import timedelta
from typing import List

import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


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


def copy_files(src_path: str, dst_path: str, filenames: List[str], compare_files: bool = False):
    logger.info("Copying files to directory...")
    if not os.path.exists(dst_path):
        os.makedirs(dst_path, exist_ok=True)
    copy_count = 0
    for file in filenames:
        src_file = os.path.join(src_path, file)
        dst_file = os.path.join(dst_path, file)
        if not os.path.exists(dst_file) or (compare_files and not filecmp.cmp(src_file, dst_file)):
            shutil.copy(src_file, dst_file)
            copy_count += 1
    logger.info(f"{copy_count} files copied successfully!")


def resume_wandb_config(bash_cfg: DictConfig) -> DictConfig:
    import wandb

    wandb.login()
    api = wandb.Api()
    run = api.run(f"{bash_cfg.wandb.entity}/{bash_cfg.wandb.project}/{bash_cfg.initialize.run_id}")
    logger.info(f"Resuming training for {run.id} ...")
    cfg = DictConfig(run.config)

    cfg = OmegaConf.merge(cfg, bash_cfg)
    if rank_zero_only.rank == 0:
        logger.info(OmegaConf.to_yaml(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)))

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
            train_time_interval=timedelta(hours=1),
            save_top_k=1,
            save_last="link",
        ),
    ]
    return callbacks
