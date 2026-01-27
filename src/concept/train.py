import logging
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
from concept.utils import (
    _get_callbacks,
    copy_files,
    get_profiler,
    resume_wandb_config,
    load_pretrained_vocabulary,
    merge_lists,
    check_organism_in_h5ad_files,
)

logger = logging.getLogger(__name__)


def train(cfg: DictConfig, build_only: bool = False):
    """
    Train a model using the configuration in cfg.

    Args:
        cfg: Configuration dictionary.
    """
    # Validate configuration constraints
    scConcept.validate_config(cfg)

    val_loader_names = []
    if "val" in cfg.datamodule.dataset and cfg.datamodule.dataset.val is not None:
        val_loader_names = sorted(list(cfg.datamodule.dataset.val.keys()))


    gene_mapping = pd.read_csv(cfg.PATH.GENE_MAPPING_PATH, index_col="gene_id")["token"].to_dict()
    tokenizer = GeneIdTokenizer(gene_mapping)

    pretrained_vocabulary = None
    if "PRETRAINED_VOCABULARY" in cfg.PATH and cfg.PATH.PRETRAINED_VOCABULARY is not None:
        pretrained_vocabulary = load_pretrained_vocabulary(cfg.PATH.PRETRAINED_VOCABULARY, tokenizer)

    dataset_kwargs = OmegaConf.to_container(cfg.datamodule.dataset, resolve=True, throw_on_missing=True)
    dataloader_kwargs = OmegaConf.to_container(cfg.datamodule.dataloader, resolve=True, throw_on_missing=True)

    merged_split = set()
    if "train" in dataset_kwargs and dataset_kwargs["train"] is not None:
        dataset_kwargs["train"]["split"] = merge_lists(dataset_kwargs["train"]["split"])
        merged_split.update(dataset_kwargs["train"]["split"])
    if "val" in dataset_kwargs and dataset_kwargs["val"] is not None:
        for val_name, val_kwargs in dataset_kwargs["val"].items():
            dataset_kwargs["val"][val_name]["split"] = merge_lists(val_kwargs["split"])
            merged_split.update(dataset_kwargs["val"][val_name]["split"])

    # Copy files to faster local directory
    data_path = cfg.PATH.ADATA_PATH
    if cfg.PATH.LOCAL_DIR is not None and rank_zero_only.rank == 0:
        copy_files(
            data_path,
            cfg.PATH.LOCAL_DIR,
            list(merged_split),
            compare_files=True,
            force_copy=False,
        )
        data_path = cfg.PATH.LOCAL_DIR

    if "train" in dataset_kwargs and dataset_kwargs["train"] is not None:
        dataset_kwargs["train"]["split"] = [os.path.join(data_path, file) for file in dataset_kwargs["train"]["split"]]
    if "val" in dataset_kwargs and dataset_kwargs["val"] is not None:
        for val_name, val_kwargs in dataset_kwargs["val"].items():
            dataset_kwargs["val"][val_name]["split"] = [os.path.join(data_path, file) for file in val_kwargs["split"]]

    check_organism_in_h5ad_files([os.path.join(data_path, file) for file in merged_split])

    datamodule_args = {
        "panels_path": cfg.PATH.PANELS_PATH,
        "columns": cfg.datamodule.columns,
        "precomp_embs_key": cfg.datamodule.precomp_embs_key,
        "normalization": cfg.datamodule.normalization,
        "gene_sampling_strategy": cfg.datamodule.gene_sampling_strategy,
        "model_speed_sanity_check": cfg.datamodule.model_speed_sanity_check,
        # make sure to pass a copy to avoid being modified before uploading to wandb:
        "dataset_kwargs": dataset_kwargs,
        "dataloader_kwargs": dataloader_kwargs,
        "val_loader_names": val_loader_names,
        "tokenizer": tokenizer,
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
        wandb_logger = WandbLogger(
            name=cfg.wandb.run_name,
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            save_dir=cfg.PATH.PROJECT_PATH,
            log_model=False,
            **kwargs,
        )
        if rank_zero_only.rank == 0 and not RESUME_LOGGER:
            wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    run_id = "dummy"
    if rank_zero_only.rank == 0:
        run_id = wandb_logger.experiment.id if cfg.wandb.enabled else f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    CHECKPOINT_PATH = os.path.join(cfg.PATH.CHECKPOINT_ROOT, run_id)

    profiler = get_profiler(CHECKPOINT_PATH) if cfg.profiler.enabled else None

    trainer_kwargs = {
        "max_steps": cfg.model.training.max_steps,
        "accelerator": cfg.model.training.accelerator,
        "devices": cfg.model.training.devices,
        "num_nodes": int(os.environ["SLURM_JOB_NUM_NODES"])
        if "SLURM_JOB_NUM_NODES" in os.environ
        else cfg.model.training.num_nodes,
        "logger": wandb_logger if cfg.wandb.enabled else None,
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
        "pretrained_vocabulary": pretrained_vocabulary,
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


    if build_only:
        return trainer, model, datamodule

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    L.seed_everything(42)

    bash_cfg = OmegaConf.from_cli()

    if "initialize" in bash_cfg and bash_cfg.initialize.resume:
        cfg = resume_wandb_config(bash_cfg)
    else:
        logger.info("Starting new training ...")
        with initialize(version_base=None, config_path="./conf"):
            cfg = compose(config_name="config", overrides=sys.argv[1:])

    train(cfg)
