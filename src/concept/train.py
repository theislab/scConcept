import logging
import os
import sys
import datetime

import lightning as L
import pandas as pd
from hydra import compose, initialize
from concept.dataset import MultiSpeciesTokenizer
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
    resolve_split_list,
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

    local_rank = int(os.environ.get("SLURM_LOCALID", -1))
    global_rank = rank_zero_only.rank
    logger.info(f"GLOBAL_RANK: {global_rank}, LOCAL_RANK: {local_rank}")

    val_loader_names = []
    if "val" in cfg.datamodule.dataset and cfg.datamodule.dataset.val is not None:
        val_loader_names = sorted(list(cfg.datamodule.dataset.val.keys()))

    species_gene_mappings: dict[str, dict] = {}

    for species, mapping_path in cfg.PATH.GENE_MAPPING_PATHS.items():
        species_gene_mappings[species] = pd.read_csv(mapping_path, index_col="gene_id")["token"].to_dict()

    tokenizer = MultiSpeciesTokenizer(species_gene_mappings)
    vocab_sizes = tokenizer.vocab_sizes

    pretrained_vocabularies = None
    if "PRETRAINED_VOCABULARY" in cfg.PATH and cfg.PATH.PRETRAINED_VOCABULARY is not None:
        pretrained_vocabularies = load_pretrained_vocabulary(cfg.PATH.PRETRAINED_VOCABULARY, tokenizer)

    if cfg.PATH.LOCAL_DIR is not None:
        for key, value in cfg.datamodule.items():
            if isinstance(value, (dict, DictConfig)) and "source_name" in value and "source_path" in value:
                source_name = value["source_name"]
                source_path = value["source_path"]
                files = list(value["train"]) + list(value["val"])
                if local_rank == 0:
                    copy_files(
                        source_path,
                        os.path.join(cfg.PATH.LOCAL_DIR, source_name),
                        files,
                        compare_files=True,
                        force_copy=False,
                    )
                cfg.datamodule[key]["source_path"] = os.path.join(cfg.PATH.LOCAL_DIR, source_name)

    dataset_kwargs = OmegaConf.to_container(cfg.datamodule.dataset, resolve=True, throw_on_missing=True)
    dataloader_kwargs = OmegaConf.to_container(cfg.datamodule.dataloader, resolve=True, throw_on_missing=True)

    if "train" in dataset_kwargs and dataset_kwargs["train"] is not None:
        paths, metadata = resolve_split_list(dataset_kwargs["train"]["split"], key="train")
        dataset_kwargs["train"]["split"] = {"paths": paths, "metadata": metadata}
    if "val" in dataset_kwargs and dataset_kwargs["val"] is not None:
        for val_name, val_kwargs in dataset_kwargs["val"].items():
            paths, metadata = resolve_split_list(val_kwargs["split"], key="val")
            dataset_kwargs["val"][val_name]["split"] = {"paths": paths, "metadata": metadata}

    datamodule_args = {
        "panels_path": cfg.PATH.PANELS_PATH,
        "obs_keys": cfg.datamodule.obs_keys,
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
        if global_rank == 0 and not RESUME_LOGGER:
            wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    run_id = "dummy"
    if global_rank == 0:
        run_id = (
            wandb_logger.experiment.id
            if cfg.wandb.enabled
            else f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )

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
        "gradient_clip_val": cfg.model.training.gradient_clip_val,
        "profiler": profiler,
        "callbacks": _get_callbacks(CHECKPOINT_PATH, cfg.model.training.max_steps),
    }
    trainer = L.Trainer(
        **trainer_kwargs,
        strategy=DDPStrategy(find_unused_parameters=True, skip_all_reduce_unused_params=True),
        precision="bf16-mixed",
        use_distributed_sampler=False,
    )

    model_args = {
        "config": cfg.model,
        "pad_token_id": tokenizer.PAD_TOKEN,
        "cls_token_id": tokenizer.CLS_TOKEN,
        "vocab_sizes": vocab_sizes,
        "pretrained_vocabularies": pretrained_vocabularies,
        "world_size": trainer.world_size,
        "val_loader_names": val_loader_names,
        "obs_keys": cfg.datamodule.obs_keys,
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
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, hard))

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
