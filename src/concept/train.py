import os
import sys
from datetime import datetime

import lightning as L
import pandas as pd
from hydra import compose, initialize
from lamin_dataloader import GeneIdTokenizer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from concept import ContrastiveModel, scConcept
from concept.data import AnnDataModule
from concept.utils import _get_callbacks, copy_files, get_profiler, resume_wandb_config


def train(cfg: DictConfig):
    """
    Train a model using the configuration in cfg.

    Args:
        cfg: Configuration dictionary.
    """
    # Validate configuration constraints
    scConcept.validate_config(cfg)

    # Copy files to faster local directory
    data_path = cfg.PATH.ADATA_PATH
    if cfg.PATH.LOCAL_DIR is not None and rank_zero_only.rank == 0:
        copy_files(
            data_path,
            cfg.PATH.LOCAL_DIR,
            list(cfg.PATH.SPLIT.get("train", [])) + list(cfg.PATH.SPLIT.get("val", [])),
            compare_files=False,
        )
        data_path = cfg.PATH.LOCAL_DIR

    if "val" in cfg.datamodule.dataset and cfg.datamodule.dataset.val is not None:
        val_loader_names = sorted(list(cfg.datamodule.dataset.val.keys()))
    else:
        val_loader_names = []

    gene_mapping = pd.read_pickle(cfg.PATH.gene_mapping_path).to_dict()

    split = {}
    for key, filenames in cfg.PATH.SPLIT.items():
        if filenames is not None:
            split[key] = [os.path.join(data_path, file) for file in filenames]

    datamodule_args = {
        "split": split,
        "panels_path": cfg.PATH.PANELS_PATH,
        "columns": cfg.datamodule.columns,
        "precomp_embs_key": cfg.datamodule.precomp_embs_key,
        "normalization": cfg.datamodule.normalization,
        "gene_sampling_strategy": cfg.datamodule.gene_sampling_strategy,
        "model_speed_sanity_check": cfg.datamodule.model_speed_sanity_check,
        # make sure to pass a copy to avoid being modified before uploading to wandb:
        "dataset_kwargs": {**OmegaConf.to_container(cfg.datamodule.dataset, resolve=True, throw_on_missing=True)},
        "dataloader_kwargs": {**OmegaConf.to_container(cfg.datamodule.dataloader, resolve=True, throw_on_missing=True)},
        "val_loader_names": val_loader_names,
        "tokenizer": GeneIdTokenizer(gene_mapping),
    }
    datamodule = AnnDataModule(**datamodule_args)

    RESUME_LOGGER = cfg.initialize.resume and not cfg.initialize.create_new_run
    if cfg.wandb.enabled:
        if cfg.wandb.entity is None or cfg.wandb.project is None or cfg.wandb.run_name is None:
            raise ValueError("wandb.entity, wandb.project, and wandb.run_name are required when wandb.enabled is True")
        kwargs = {}
        if RESUME_LOGGER:
            kwargs = {
                "id": cfg.initialize.run_id,
                "resume": "allow",
                "tags": os.environ.get("WANDB_TAGS", "").split(","),
            }
        logger = WandbLogger(
            name=cfg.wandb.run_name,
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            save_dir=cfg.PATH.PROJECT_PATH,
            log_model=False,
            **kwargs,
        )
        if rank_zero_only.rank == 0 and not RESUME_LOGGER:
            logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    run_id = "dummy"
    if rank_zero_only.rank == 0:
        run_id = logger.experiment.id if cfg.wandb.enabled else f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    CHECKPOINT_PATH = os.path.join(cfg.PATH.CHECKPOINT_ROOT, run_id)

    profiler = get_profiler(CHECKPOINT_PATH) if cfg.profiler.enabled else None

    trainer_kwargs = {
        "max_steps": cfg.model.training.max_steps,
        "accelerator": cfg.model.training.accelerator,
        "devices": cfg.model.training.devices,
        "num_nodes": int(os.environ["SLURM_JOB_NUM_NODES"])
        if "SLURM_JOB_NUM_NODES" in os.environ
        else cfg.model.training.num_nodes,
        "logger": logger if cfg.wandb.enabled else None,
        "log_every_n_steps": cfg.model.training.log_every_n_steps,
        "val_check_interval": cfg.model.training.val_check_interval,
        "check_val_every_n_epoch": cfg.model.training.check_val_every_n_epoch,
        "limit_train_batches": cfg.model.training.limit_train_batches,
        "accumulate_grad_batches": cfg.model.training.accumulate_grad_batches,
        "profiler": profiler,
        "callbacks": _get_callbacks(CHECKPOINT_PATH, cfg.model.training.max_steps),
    }
    trainer = L.Trainer(
        **trainer_kwargs,
        strategy=DDPStrategy(),
        precision="bf16-mixed",
        use_distributed_sampler=False,
    )

    model_args = {
        "config": cfg.model,
        "pad_token_id": gene_mapping["<pad>"],
        "cls_token_id": gene_mapping["<cls>"],
        "vocab_size": len(gene_mapping),
        "world_size": trainer.world_size,
        "val_loader_names": val_loader_names,
        "label_keys_to_monitor": cfg.datamodule.label_keys_to_monitor,
        "batch_keys_to_monitor": cfg.datamodule.batch_keys_to_monitor,
        "precomp_embs_key": cfg.datamodule.precomp_embs_key,
    }

    ckpt_path = None

    if cfg.initialize.resume:
        checkpoint_file = os.path.join(cfg.PATH.CHECKPOINT_ROOT, cfg.initialize.run_id, cfg.initialize.checkpoint)
        if cfg.initialize.create_new_run:
            model = ContrastiveModel.load_from_checkpoint(checkpoint_file, **model_args, strict=False)
        else:
            model = ContrastiveModel(**model_args)
            ckpt_path = checkpoint_file
    else:
        model = ContrastiveModel(**model_args)
        if cfg.model.training.validate_before_training:
            trainer.validate(model=model, datamodule=datamodule)

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    L.seed_everything(42)
    
    bash_cfg = OmegaConf.from_cli()

    if "initialize" in bash_cfg and bash_cfg.initialize.resume:
        cfg = resume_wandb_config(bash_cfg)
    else:
        print(f"Starting new training ...")
        print("overrides:", sys.argv[1:])
        with initialize(version_base=None, config_path="./conf"):
            cfg = compose(config_name="config", overrides=sys.argv[1:])

    train(cfg)
