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
from concept.utils import SLURMEnv, copy_files, load_pretrained_vocabulary, resolve_split_list, resume_wandb_config

logger = logging.getLogger(__name__)


class _FabricContrastiveModel(ContrastiveModel):
    """Thin adapter so ContrastiveModel works without a Lightning Trainer."""

    @property
    def global_step(self) -> int:
        return getattr(self, "_fabric_global_step", 0)

    @property
    def logger(self):
        return getattr(self, "_fabric_logger", None)

    def log(self, name: str, value, on_step: bool = False, on_epoch: bool = True, **kwargs) -> None:
        """Route logging based on the current stage and on_step/on_epoch flags.

        During training, ``on_step=True`` metrics are forwarded to Fabric
        immediately; ``on_epoch=True``-only metrics are skipped (no epoch
        aggregation in the Fabric loop).

        During validation, ``on_epoch=True`` metrics are accumulated in
        ``_val_metric_accumulator`` so they can be averaged and flushed
        after the full pass.
        """
        fabric: Fabric | None = getattr(self, "_fabric_ref", None)
        if fabric is None:
            return

        if self.stage == "train":
            if on_step:
                v = value.item() if hasattr(value, "item") else float(value)
                fabric.log(name, v, step=self._fabric_global_step)
            # on_epoch during training is skipped – no epoch aggregation in the Fabric loop
        elif self.stage == "val":
            if on_epoch:
                acc: dict | None = getattr(self, "_val_metric_accumulator", None)
                if acc is not None:
                    v = value.item() if hasattr(value, "item") else float(value)
                    acc.setdefault(name, []).append(v)
            # on_step during validation is skipped

    def log_dict(self, metrics: dict, on_step: bool = True, on_epoch: bool = False, **kwargs) -> None:
        """Dispatch each metric through ``log`` so stage-routing applies."""
        for name, value in metrics.items():
            self.log(name, value, on_step=on_step, on_epoch=on_epoch, **kwargs)


class FabricTrainer:
    """Minimal Fabric-based trainer that replaces the Lightning Trainer."""

    def __init__(self, cfg: DictConfig) -> None:
        scConcept.validate_config(cfg)
        self.cfg = cfg
        self.global_step = 0
        self.epoch = 0

        self.tokenizer = self._build_tokenizer()
        self.pretrained_vocabularies = self._load_pretrained_vocabularies()
        self.fabric = self._build_fabric()
        self.fabric.launch()
        self._upload_wandb_config()
        self.checkpoint_path = self._resolve_checkpoint_path()
        self._save_config()
        self.profiler = self._build_profiler()
        self._copy_datasets_to_local()
        self.datamodule = self._build_datamodule()
        self.model, self.optimizer, self.scheduler = self._build_model_and_optimizer()
        self.train_dataloader = self.fabric.setup_dataloaders(
            self.datamodule.train_dataloader(), use_distributed_sampler=False
        )
        self.val_dataloaders = self._setup_val_dataloaders()
        self._resume_if_requested()
        self.max_steps = cfg.model.training.max_steps
        self.accumulate_grad_batches = cfg.model.training.accumulate_grad_batches
        self.log_every_n_steps = cfg.model.training.log_every_n_steps
        self.gradient_clip_val = cfg.model.training.gradient_clip_val
        self.val_check_interval = cfg.model.training.val_check_interval
        self.data_loading_speed_sanity_check = cfg.model.data_loading_speed_sanity_check

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_tokenizer(self) -> MultiSpeciesTokenizer:
        species_gene_mappings = {
            species: pd.read_csv(os.path.join(self.cfg.PATH.GENE_MAPPINGS_PATH, f"{species}.csv"), index_col="gene_id")["token"].to_dict()
            for species in self.cfg.PATH.SPECIES
        }
        return MultiSpeciesTokenizer(species_gene_mappings)

    def _load_pretrained_vocabularies(self):
        if "PRETRAINED_VOCABULARY" in self.cfg.PATH and self.cfg.PATH.PRETRAINED_VOCABULARY is not None:
            return load_pretrained_vocabulary(self.cfg.PATH.PRETRAINED_VOCABULARY, self.tokenizer)
        return None

    def _get_logger(self):
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

        return wandb_logger

    def _upload_wandb_config(self):
        if self.cfg.wandb.enabled and self.fabric.is_global_zero and not self._resume_logger:
            self.fabric.logger.experiment.config.update(
                OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True)
            )

    def _build_profiler(self):
        """Create a torch.profiler.profile instance when profiling is enabled."""
        if not self.cfg.profiler.enabled:
            return None
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                skip_first=100,
                wait=10,
                warmup=10,
                active=20,
                repeat=1,
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )

    def _build_fabric(self) -> Fabric:
        logger = self._get_logger()
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", self.cfg.model.training.num_nodes))
        strategy = DDPStrategy(find_unused_parameters=True, skip_all_reduce_unused_params=True) if len(self.tokenizer.species) > 1 else DDPStrategy()
        return Fabric(
            accelerator=self.cfg.model.training.accelerator,
            devices=self.cfg.model.training.devices,
            num_nodes=num_nodes,
            strategy=strategy,
            precision="bf16-mixed",
            loggers=logger,
            plugins=[SLURMEnv()] if os.environ.get("SLURM_JOB_ID", None) is not None else None,
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

    def _save_config(self) -> None:
        """Persist the resolved config as YAML to <checkpoint_path>/config/config.yaml."""
        if not self.fabric.is_global_zero:
            return
        os.makedirs(self.checkpoint_path, exist_ok=True)
        config_path = os.path.join(self.checkpoint_path, "config.yaml")
        OmegaConf.save(self.cfg, config_path)
        logger.info("Saved resolved config to %s", config_path)

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
                        compare_files=True,
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

        model._fabric_ref = self.fabric
        if self.cfg.wandb.enabled:
            model._fabric_logger = self.fabric.logger

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
            remainder = self.fabric.load(checkpoint_file, {"model": self.model, "optimizer": self.optimizer, "datamodule": self.datamodule})
            self.global_step = int(remainder.get("global_step", 0))
            self.epoch = int(remainder.get("epoch", 0))
            logger.info("Resumed from %s at step %d, epoch %d", checkpoint_file, self.global_step, self.epoch)

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

        for loader_idx, (val_name, val_loader) in enumerate(
            zip(self.val_loader_names, self.val_dataloaders, strict=False)
        ):
            self.model._val_metric_accumulator = {}

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    self.model.validation_step(batch, batch_idx, dataloader_idx=loader_idx)

            accumulated = self.model._val_metric_accumulator
            if accumulated:
                averaged = {name: sum(vals) / len(vals) for name, vals in accumulated.items()}
                self.fabric.log_dict(averaged, step=self.global_step)
                if self.fabric.is_global_zero:
                    summary = {k: f"{v:.4f}" for k, v in averaged.items() if "loss" in k or "recall" in k}
                    logger.info("val [step=%d] %s: %s", self.global_step, val_name, summary)

        self.model._val_metric_accumulator = None

        with torch.no_grad():
            self.model.on_validation_epoch_end()

        self.model.train()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, label: str) -> None:
        state = {"model": self.model, "optimizer": self.optimizer, "global_step": self.global_step, "epoch": self.epoch, "datamodule": self.datamodule}
        path = os.path.join(self.checkpoint_path, label)
        self.fabric.save(path, state)

    def _set_sampler_epoch(self, epoch: int) -> None:
        """Forward the new epoch to the train dataloader's sampler so it generates fresh samples."""
        sampler = getattr(self.train_dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def _maybe_checkpoint(self) -> None:
        milestone_interval = 100_000 if self.max_steps > 100_000 else 10_000
        latest_interval = 20_000

        if self.global_step % milestone_interval == 0:
            label = os.path.join("steps", f"step={self.global_step}.ckpt")
            self._save_checkpoint(label)
            logger.info("Saved milestone checkpoint at step %d", self.global_step)

        if self.global_step % latest_interval == 0:
            self._save_checkpoint(os.path.join("latest", "last.ckpt"))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _log_parameter_summary(self) -> None:
        """Log the number of trainable and non-trainable model parameters (rank 0 only)."""
        if not self.fabric.is_global_zero:
            return
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        logger.info(
            "Model parameters — trainable: %s | non-trainable: %s | total: %s",
            f"{trainable:,}",
            f"{non_trainable:,}",
            f"{trainable + non_trainable:,}",
        )

    def fit(self) -> None:
        self.model.on_fit_start()
        """Run the full training loop."""
        self._log_parameter_summary()

        if self.cfg.model.training.validate_before_training and not self.cfg.initialize.resume:
            logger.info("Running validation before training ...")
            self._run_validation()

        self.model.train()
        last_logging_step = self.global_step
        last_log_time = time.perf_counter()

        if self.profiler is not None:
            self.profiler.start()


        while True:
            if self.global_step >= self.max_steps:
                break
            self.train_dataloader._num_iter_calls = self.epoch
            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(self.train_dataloader):
                if self.global_step >= self.max_steps:
                    break

                self.model._fabric_global_step = self.global_step
                is_accumulating = (batch_idx + 1) % self.accumulate_grad_batches != 0

                with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                    loss = self.model.training_step(batch, batch_idx)
                    if not self.data_loading_speed_sanity_check:
                        self.fabric.backward(loss / self.accumulate_grad_batches)

                if not is_accumulating:
                    self.model.on_before_optimizer_step(self.optimizer)
                    if self.gradient_clip_val:
                        self.fabric.clip_gradients(self.model, self.optimizer, max_norm=self.gradient_clip_val, error_if_nonfinite=False)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.global_step += 1
                    if self.profiler is not None:
                        self.profiler.step()
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
                    lr_per_group = {
                        f"trainer/lr_pg_{i}": pg["lr"]
                        for i, pg in enumerate(self.optimizer.param_groups)
                    }
                    logger.info(
                        "epoch=%d, step=%d  loss=%.4f  speed=%.2f batches/s",
                        self.epoch, self.global_step, loss.item(), batches_per_sec,
                    )
                    self.fabric.log_dict(
                        {"trainer/batches_per_sec": batches_per_sec, "trainer/epoch": self.epoch, **lr_per_group},
                        step=self.global_step,
                    )
                    last_logging_step = self.global_step
                    last_log_time = now

            self.epoch += 1
            self._set_sampler_epoch(self.epoch)

        if self.profiler is not None:
            self.profiler.stop()
            trace_path = os.path.join(
                self.checkpoint_path, "profiler",
                f"chrome_trace_rank{self.fabric.global_rank}.json",
            )
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            self.profiler.export_chrome_trace(trace_path)
            logger.info("Profiler chrome trace saved to %s", trace_path)

        logger.info("Training complete at step %d", self.global_step)


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
