import datetime
import logging
import os
import sys
import time

import lightning as L
import pandas as pd
from hydra import compose, initialize
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from wandb.integration.lightning.fabric import WandbLogger

from concept import ContrastiveModel, scConcept
from concept.data import AnnDataModule
from concept.dataset import MultiSpeciesTokenizer
from concept.utils import copy_files, load_pretrained_vocabulary, resolve_split_list

logger = logging.getLogger(__name__)


class _FabricContrastiveModel(ContrastiveModel):
    """Thin adapter so ContrastiveModel works without a Lightning Trainer."""

    @property
    def global_step(self) -> int:
        return getattr(self, "_fabric_global_step", 0)

    @property
    def logger(self):
        return getattr(self, "_fabric_logger", None)

    def log(self, *args, **kwargs) -> None:
        pass

    def log_dict(self, *args, **kwargs) -> None:
        pass


def train(cfg: DictConfig) -> None:
    """Train using Lightning Fabric instead of the full Lightning Trainer."""
    scConcept.validate_config(cfg)

    # Build tokenizer
    species_gene_mappings: dict = {}
    for species, mapping_path in cfg.PATH.GENE_MAPPING_PATHS.items():
        species_gene_mappings[species] = pd.read_csv(mapping_path, index_col="gene_id")["token"].to_dict()
    tokenizer = MultiSpeciesTokenizer(species_gene_mappings)

    pretrained_vocabularies = None
    if "PRETRAINED_VOCABULARY" in cfg.PATH and cfg.PATH.PRETRAINED_VOCABULARY is not None:
        pretrained_vocabularies = load_pretrained_vocabulary(cfg.PATH.PRETRAINED_VOCABULARY, tokenizer)

    # Initialise Fabric early so distributed is set up before dataloader creation,
    # which needs world_size for per-replica batch size calculation.
    RESUME_LOGGER = cfg.initialize.resume and not cfg.initialize.create_new_run
    wandb_logger = None
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

    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", cfg.model.training.num_nodes))
    fabric = Fabric(
        accelerator=cfg.model.training.accelerator,
        devices=cfg.model.training.devices,
        num_nodes=num_nodes,
        strategy=DDPStrategy(find_unused_parameters=True, skip_all_reduce_unused_params=True),
        precision="bf16-mixed",
        loggers=wandb_logger if wandb_logger is not None else [],
    )
    fabric.launch()

    if cfg.wandb.enabled and fabric.is_global_zero and not RESUME_LOGGER:
        fabric.logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    # Derive a stable run ID so checkpoints are grouped under one directory.
    run_id = (
        fabric.logger.experiment.id
        if cfg.wandb.enabled
        else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ) if fabric.is_global_zero else ""
    run_id = fabric.broadcast(run_id)
    CHECKPOINT_PATH = os.path.join(cfg.PATH.CHECKPOINT_ROOT, run_id)

    # Copy datasets to a fast local directory (e.g. NVMe scratch) if configured.
    # Only rank-0 on each node performs the copy; all ranks update the source_path.
    if cfg.PATH.LOCAL_DIR is not None:
        for key, value in cfg.datamodule.items():
            if isinstance(value, (dict, DictConfig)) and "source_name" in value and "source_path" in value:
                source_name = value["source_name"]
                source_path = value["source_path"]
                files = list(value["train"]) + list(value["val"])
                if fabric.is_global_zero:
                    copy_files(
                        source_path,
                        os.path.join(cfg.PATH.LOCAL_DIR, source_name),
                        files,
                        compare_files=False,
                        force_copy=False,
                    )
                cfg.datamodule[key]["source_path"] = os.path.join(cfg.PATH.LOCAL_DIR, source_name)
        fabric.barrier()  # wait for rank-0 to finish copying before other ranks open the files

    # Resolve dataset splits
    val_loader_names = []
    if "val" in cfg.datamodule.dataset and cfg.datamodule.dataset.val is not None:
        val_loader_names = sorted(cfg.datamodule.dataset.val.keys())

    dataset_kwargs = OmegaConf.to_container(cfg.datamodule.dataset, resolve=True, throw_on_missing=True)
    dataloader_kwargs = OmegaConf.to_container(cfg.datamodule.dataloader, resolve=True, throw_on_missing=True)

    if "train" in dataset_kwargs and dataset_kwargs["train"] is not None:
        paths, metadata = resolve_split_list(dataset_kwargs["train"]["split"], key="train")
        dataset_kwargs["train"]["split"] = {"paths": paths, "metadata": metadata}
    if "val" in dataset_kwargs and dataset_kwargs["val"] is not None:
        for val_name, val_kwargs in dataset_kwargs["val"].items():
            paths, metadata = resolve_split_list(val_kwargs["split"], key="val")
            dataset_kwargs["val"][val_name]["split"] = {"paths": paths, "metadata": metadata}

    datamodule = AnnDataModule(
        panels_path=cfg.PATH.PANELS_PATH,
        obs_keys=cfg.datamodule.obs_keys,
        precomp_embs_key=cfg.datamodule.precomp_embs_key,
        normalization=cfg.datamodule.normalization,
        gene_sampling_strategy=cfg.datamodule.gene_sampling_strategy,
        model_speed_sanity_check=cfg.datamodule.model_speed_sanity_check,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
        val_loader_names=val_loader_names,
        tokenizer=tokenizer,
    )

    # Build model and optimizer
    model = _FabricContrastiveModel(
        config=cfg.model,
        pad_token_id=tokenizer.PAD_TOKEN,
        cls_token_id=tokenizer.CLS_TOKEN,
        vocab_sizes=tokenizer.vocab_sizes,
        pretrained_vocabularies=pretrained_vocabularies,
        world_size=fabric.world_size,
        val_loader_names=val_loader_names,
        obs_keys=cfg.datamodule.obs_keys,
        precomp_embs_key=cfg.datamodule.precomp_embs_key,
    )

    if cfg.model.data_loading_speed_sanity_check:
        model.requires_grad_(False)


    if cfg.wandb.enabled:
        model._fabric_logger = fabric.logger
        model.log = lambda name, value, **kwargs: fabric.log(name, value, step=model._fabric_global_step)
        model.log_dict = lambda metrics, **kwargs: fabric.log_dict(metrics, step=model._fabric_global_step)


    opt_result = model.configure_optimizers()
    if isinstance(opt_result, tuple):
        (optimizer,), (scheduler_cfg,) = opt_result
        scheduler = scheduler_cfg["scheduler"]
    else:
        optimizer = opt_result
        scheduler = None

    model, optimizer = fabric.setup(model, optimizer)

    # Resume from checkpoint if requested.
    global_step = 0
    if cfg.initialize.resume:
        checkpoint_file = os.path.join(
            cfg.PATH.CHECKPOINT_ROOT, cfg.initialize.run_id, cfg.initialize.checkpoint
        )
        if cfg.initialize.create_new_run:
            # Load model weights only (fine-tuning / new W&B run), allow partial match.
            fabric.load(checkpoint_file, {"model": model}, strict=False)
            logger.info("Loaded model weights from %s (new run, strict=False)", checkpoint_file)
        else:
            # Full resume: restore model, optimizer and the step counter.
            remainder = fabric.load(checkpoint_file, {"model": model, "optimizer": optimizer})
            global_step = int(remainder.get("global_step", 0))
            logger.info("Resumed from %s at step %d", checkpoint_file, global_step)

    # AnnDataModule already adds DistributedSamplerWrapper when dist is initialised,
    # so we skip Fabric's built-in distributed sampler to avoid double-wrapping.
    train_dataloader = fabric.setup_dataloaders(
        datamodule.train_dataloader(), use_distributed_sampler=False
    )

    # Checkpoint intervals (mirrors _get_callbacks in utils.py):
    #   steps/   — milestone snapshots kept forever
    #   latest/  — rolling "latest" overwritten every 20k steps
    max_steps = cfg.model.training.max_steps
    milestone_interval = 100_000 if max_steps > 100_000 else 10_000
    latest_interval = 20_000

    # Training loop
    model.train()
    last_logging_step = 0
    accumulate_grad_batches = cfg.model.training.accumulate_grad_batches
    log_every_n_steps = cfg.model.training.log_every_n_steps
    gradient_clip_val = cfg.model.training.gradient_clip_val
    last_log_time = time.perf_counter()

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_dataloader):
        if global_step >= max_steps:
            break

        model._fabric_global_step = global_step
        is_accumulating = (batch_idx + 1) % accumulate_grad_batches != 0

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            loss = model.training_step(batch, batch_idx)
            fabric.backward(loss / accumulate_grad_batches)

        if not is_accumulating:
            if gradient_clip_val:
                fabric.clip_gradients(model, optimizer, max_norm=gradient_clip_val)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            ckpt_state = {"model": model, "optimizer": optimizer, "global_step": global_step}

            # Milestone checkpoint — kept forever.
            if global_step % milestone_interval == 0:
                path = os.path.join(CHECKPOINT_PATH, "steps", f"step={global_step}.ckpt")
                fabric.save(path, ckpt_state)
                logger.info("Saved milestone checkpoint: %s", path)

            # Rolling latest checkpoint — overwrites previous.
            if global_step % latest_interval == 0:
                fabric.save(os.path.join(CHECKPOINT_PATH, "latest", "latest.ckpt"), ckpt_state)

        if fabric.is_global_zero and global_step % log_every_n_steps == 0 and global_step != last_logging_step:
            now = time.perf_counter()
            batches_per_sec = (global_step - last_logging_step) / (now - last_log_time)
            logger.info("step=%d  loss=%.4f  speed=%.2f batches/s", global_step, loss.item(), batches_per_sec)
            last_logging_step = global_step
            last_log_time = now


if __name__ == "__main__":
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, hard))

    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    L.seed_everything(42)

    with initialize(version_base=None, config_path="./conf"):
        cfg = compose(config_name="config", overrides=sys.argv[1:])

    train(cfg)
