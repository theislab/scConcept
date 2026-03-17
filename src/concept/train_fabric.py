import datetime
import logging
import os
import sys
import time

import lightning as L
import pandas as pd
import torch
from hydra import compose, initialize
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from wandb.integration.lightning.fabric import WandbLogger

from concept import ContrastiveModel, scConcept
from concept.data import AnnDataModule
from concept.dataset import MultiSpeciesTokenizer
from concept.utils import copy_files, load_pretrained_vocabulary, resolve_split_list, resume_wandb_config

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


class FabricTrainer:
    """Minimal Fabric-based trainer that replaces the Lightning Trainer."""

    def __init__(self, cfg: DictConfig) -> None:
        scConcept.validate_config(cfg)
        self.cfg = cfg
        self.global_step = 0

        self.tokenizer = self._build_tokenizer()
        self.pretrained_vocabularies = self._load_pretrained_vocabularies()
        self.fabric = self._build_fabric()
        self.fabric.launch()
        self._upload_wandb_config()
        self.checkpoint_path = self._resolve_checkpoint_path()
        self._copy_datasets_to_local()
        self.datamodule = self._build_datamodule()
        self.model, self.optimizer, self.scheduler = self._build_model_and_optimizer()
        self._resume_if_requested()
        self.train_dataloader = self.fabric.setup_dataloaders(
            self.datamodule.train_dataloader(), use_distributed_sampler=False
        )
        self.val_dataloaders = self._setup_val_dataloaders()
        self.max_steps = cfg.model.training.max_steps
        self.accumulate_grad_batches = cfg.model.training.accumulate_grad_batches
        self.log_every_n_steps = cfg.model.training.log_every_n_steps
        self.gradient_clip_val = cfg.model.training.gradient_clip_val
        self.val_check_interval = cfg.model.training.val_check_interval

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_tokenizer(self) -> MultiSpeciesTokenizer:
        species_gene_mappings = {
            species: pd.read_csv(path, index_col="gene_id")["token"].to_dict()
            for species, path in self.cfg.PATH.GENE_MAPPING_PATHS.items()
        }
        return MultiSpeciesTokenizer(species_gene_mappings)

    def _load_pretrained_vocabularies(self):
        if "PRETRAINED_VOCABULARY" in self.cfg.PATH and self.cfg.PATH.PRETRAINED_VOCABULARY is not None:
            return load_pretrained_vocabulary(self.cfg.PATH.PRETRAINED_VOCABULARY, self.tokenizer)
        return None

    def _build_fabric(self) -> Fabric:
        resume_logger = self.cfg.initialize.resume and not self.cfg.initialize.create_new_run

        wandb_logger = None
        if self.cfg.wandb.enabled:
            if self.cfg.wandb.entity is None or self.cfg.wandb.project is None or self.cfg.wandb.run_name is None:
                raise ValueError(
                    "wandb.entity, wandb.project, and wandb.run_name are required when wandb.enabled is True"
                )
            kwargs = (
                {
                    "id": self.cfg.initialize.run_id,
                    "resume": "allow",
                    "tags": os.environ.get("WANDB_TAGS", "").split(","),
                }
                if resume_logger
                else {}
            )
            wandb_logger = WandbLogger(
                name=self.cfg.wandb.run_name,
                entity=self.cfg.wandb.entity,
                project=self.cfg.wandb.project,
                save_dir=self.cfg.PATH.PROJECT_PATH,
                log_model=False,
                **kwargs,
            )

        self._resume_logger = resume_logger
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", self.cfg.model.training.num_nodes))
        return Fabric(
            accelerator=self.cfg.model.training.accelerator,
            devices=self.cfg.model.training.devices,
            num_nodes=num_nodes,
            strategy=DDPStrategy(find_unused_parameters=True, skip_all_reduce_unused_params=True),
            precision="bf16-mixed",
            loggers=wandb_logger if wandb_logger is not None else [],
        )

    def _upload_wandb_config(self) -> None:
        if self.cfg.wandb.enabled and self.fabric.is_global_zero and not self._resume_logger:
            self.fabric.logger.experiment.config.update(
                OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True)
            )

    def _resolve_checkpoint_path(self) -> str:
        run_id = (
            (
                self.fabric.logger.experiment.id
                if self.cfg.wandb.enabled
                else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
            if self.fabric.is_global_zero
            else ""
        )
        run_id = self.fabric.broadcast(run_id)
        return os.path.join(self.cfg.PATH.CHECKPOINT_ROOT, run_id)

    def _copy_datasets_to_local(self) -> None:
        """Copy dataset files to a fast local directory (e.g. NVMe scratch) if configured."""
        if self.cfg.PATH.LOCAL_DIR is None:
            return
        for key, value in self.cfg.datamodule.items():
            if isinstance(value, (dict, DictConfig)) and "source_name" in value and "source_path" in value:
                source_name = value["source_name"]
                source_path = value["source_path"]
                files = list(value["train"]) + list(value["val"])
                if self.fabric.local_rank == 0:
                    copy_files(
                        source_path,
                        os.path.join(self.cfg.PATH.LOCAL_DIR, source_name),
                        files,
                        compare_files=False,
                        force_copy=False,
                    )
                self.cfg.datamodule[key]["source_path"] = os.path.join(self.cfg.PATH.LOCAL_DIR, source_name)
        self.fabric.barrier()

    def _build_datamodule(self) -> AnnDataModule:
        val_loader_names = []
        if "val" in self.cfg.datamodule.dataset and self.cfg.datamodule.dataset.val is not None:
            val_loader_names = sorted(self.cfg.datamodule.dataset.val.keys())
        self.val_loader_names = val_loader_names

        dataset_kwargs = OmegaConf.to_container(self.cfg.datamodule.dataset, resolve=True, throw_on_missing=True)
        dataloader_kwargs = OmegaConf.to_container(self.cfg.datamodule.dataloader, resolve=True, throw_on_missing=True)

        if "train" in dataset_kwargs and dataset_kwargs["train"] is not None:
            paths, metadata = resolve_split_list(dataset_kwargs["train"]["split"], key="train")
            dataset_kwargs["train"]["split"] = {"paths": paths, "metadata": metadata}
        if "val" in dataset_kwargs and dataset_kwargs["val"] is not None:
            for val_name, val_kwargs in dataset_kwargs["val"].items():
                paths, metadata = resolve_split_list(val_kwargs["split"], key="val")
                dataset_kwargs["val"][val_name]["split"] = {"paths": paths, "metadata": metadata}

        return AnnDataModule(
            panels_path=self.cfg.PATH.PANELS_PATH,
            obs_keys=self.cfg.datamodule.obs_keys,
            precomp_embs_key=self.cfg.datamodule.precomp_embs_key,
            normalization=self.cfg.datamodule.normalization,
            gene_sampling_strategy=self.cfg.datamodule.gene_sampling_strategy,
            model_speed_sanity_check=self.cfg.datamodule.model_speed_sanity_check,
            dataset_kwargs=dataset_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            val_loader_names=self.val_loader_names,
            tokenizer=self.tokenizer,
        )

    def _build_model_and_optimizer(self):
        model = _FabricContrastiveModel(
            config=self.cfg.model,
            pad_token_id=self.tokenizer.PAD_TOKEN,
            cls_token_id=self.tokenizer.CLS_TOKEN,
            vocab_sizes=self.tokenizer.vocab_sizes,
            pretrained_vocabularies=self.pretrained_vocabularies,
            world_size=self.fabric.world_size,
            val_loader_names=self.val_loader_names,
            obs_keys=self.cfg.datamodule.obs_keys,
            precomp_embs_key=self.cfg.datamodule.precomp_embs_key,
        )

        if self.cfg.model.data_loading_speed_sanity_check:
            model.requires_grad_(False)

        if self.cfg.wandb.enabled:
            model._fabric_logger = self.fabric.logger
            model.log = lambda name, value, **kw: self.fabric.log(name, value, step=model._fabric_global_step)
            model.log_dict = lambda metrics, **kw: self.fabric.log_dict(metrics, step=model._fabric_global_step)

        opt_result = model.configure_optimizers()
        if isinstance(opt_result, tuple):
            (optimizer,), (scheduler_cfg,) = opt_result
            scheduler = scheduler_cfg["scheduler"]
        else:
            optimizer = opt_result
            scheduler = None

        model, optimizer = self.fabric.setup(model, optimizer)
        return model, optimizer, scheduler

    def _resume_if_requested(self) -> None:
        if not self.cfg.initialize.resume:
            return
        checkpoint_file = os.path.join(
            self.cfg.PATH.CHECKPOINT_ROOT, self.cfg.initialize.run_id, self.cfg.initialize.checkpoint
        )
        if self.cfg.initialize.create_new_run:
            self.fabric.load(checkpoint_file, {"model": self.model}, strict=False)
            logger.info("Loaded model weights from %s (new run, strict=False)", checkpoint_file)
        else:
            remainder = self.fabric.load(checkpoint_file, {"model": self.model, "optimizer": self.optimizer})
            self.global_step = int(remainder.get("global_step", 0))
            logger.info("Resumed from %s at step %d", checkpoint_file, self.global_step)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _setup_val_dataloaders(self) -> list:
        """Wrap all validation dataloaders with Fabric."""
        if not self.val_loader_names:
            return []
        raw_loaders = self.datamodule.val_dataloader()
        wrapped = self.fabric.setup_dataloaders(*raw_loaders, use_distributed_sampler=False)
        return list(wrapped) if isinstance(wrapped, (list, tuple)) else [wrapped]

    def _run_validation(self) -> None:
        """Run a full validation pass over all configured val dataloaders."""
        if not self.val_loader_names:
            return

        self.model.eval()
        self.model._fabric_global_step = self.global_step

        def _make_collector(acc: dict) -> callable:
            def _collect(name, value, **kw):
                v = value.item() if hasattr(value, "item") else float(value)
                acc.setdefault(name, []).append(v)
            return _collect

        for loader_idx, (val_name, val_loader) in enumerate(
            zip(self.val_loader_names, self.val_dataloaders, strict=False)
        ):
            accumulated: dict[str, list[float]] = {}
            self.model.log = _make_collector(accumulated)

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    self.model.validation_step(batch, batch_idx, dataloader_idx=loader_idx)

            if accumulated:
                averaged = {name: sum(vals) / len(vals) for name, vals in accumulated.items()}
                self.fabric.log_dict(averaged, step=self.global_step)
                if self.fabric.is_global_zero:
                    summary = {k: f"{v:.4f}" for k, v in averaged.items() if "loss" in k or "recall" in k}
                    logger.info("val [step=%d] %s: %s", self.global_step, val_name, summary)

        # Restore the original log binding
        if self.cfg.wandb.enabled:
            self.model.log = lambda name, value, **kw: self.fabric.log(
                name, value, step=self.model._fabric_global_step
            )
        else:
            self.model.log = lambda *args, **kwargs: None

        # Flush sample-stats tables to W&B and clear the buffers
        with torch.no_grad():
            self.model.on_validation_epoch_end()

        self.model.train()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, label: str) -> None:
        state = {"model": self.model, "optimizer": self.optimizer, "global_step": self.global_step}
        path = os.path.join(self.checkpoint_path, label)
        self.fabric.save(path, state)

    def _maybe_checkpoint(self) -> None:
        max_steps = self.cfg.model.training.max_steps
        milestone_interval = 100_000 if max_steps > 100_000 else 10_000
        latest_interval = 20_000

        if self.global_step % milestone_interval == 0:
            label = os.path.join("steps", f"step={self.global_step}.ckpt")
            self._save_checkpoint(label)
            logger.info("Saved milestone checkpoint at step %d", self.global_step)

        if self.global_step % latest_interval == 0:
            self._save_checkpoint(os.path.join("latest", "latest.ckpt"))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Run the full training loop."""
        if self.cfg.model.training.validate_before_training and not self.cfg.initialize.resume:
            logger.info("Running validation before training ...")
            self._run_validation()

        self.model.train()
        last_logging_step = 0
        last_log_time = time.perf_counter()
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_dataloader):
            if self.global_step >= self.max_steps:
                break

            self.model._fabric_global_step = self.global_step
            is_accumulating = (batch_idx + 1) % self.accumulate_grad_batches != 0

            with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                loss = self.model.training_step(batch, batch_idx)
                self.fabric.backward(loss / self.accumulate_grad_batches)

            if not is_accumulating:
                if self.gradient_clip_val:
                    self.fabric.clip_gradients(self.model, self.optimizer, max_norm=self.gradient_clip_val)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.global_step += 1
                self._maybe_checkpoint()
                if self.val_check_interval and self.global_step % self.val_check_interval == 0:
                    self._run_validation()

            if (
                self.fabric.is_global_zero
                and self.global_step % self.log_every_n_steps == 0
                and self.global_step != last_logging_step
            ):
                now = time.perf_counter()
                batches_per_sec = (self.global_step - last_logging_step) / (now - last_log_time)
                logger.info(
                    "step=%d  loss=%.4f  speed=%.2f batches/s",
                    self.global_step, loss.item(), batches_per_sec,
                )
                self.fabric.log_dict({"train/batches_per_sec": batches_per_sec}, step=self.global_step)
                last_logging_step = self.global_step
                last_log_time = now

        self._save_checkpoint(os.path.join("latest", "latest.ckpt"))
        logger.info("Training complete at step %d. Final checkpoint saved.", self.global_step)


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
        logger.info("Resuming training run — restoring config from W&B ...")
        cfg = resume_wandb_config(bash_cfg)
    else:
        logger.info("Starting new training run ...")
        with initialize(version_base=None, config_path="./conf"):
            cfg = compose(config_name="config", overrides=sys.argv[1:])

    FabricTrainer(cfg).fit()
