import h5py
import filecmp
import logging
import os
import shutil
from datetime import timedelta
import torch
from concept.dataset import MultiSpeciesTokenizer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def resolve_split_list(
    split_list: list, key: str | None = None
) -> tuple[list, dict[str, list]]:
    """
    Expand split config refs and flatten to a single list.

    If key is set (e.g. "train" or "val"), items that are dicts with that key
    are replaced by the key's list; otherwise items are kept as-is. Then any
    list-of-lists is flattened to a single list.

    Items can also be wrapped as ``{source: <split_config>, key: "train"|"val"}``
    to override the default key for that specific entry, e.g.:

        split:
          - ${datamodule.split_cellxgene_hsapiens}        # uses default key
          - source: ${datamodule.split_cellxgene_mmusculus}
            key: val                                       # override to val sublist

    Returns:
        A tuple ``(file_paths, metadata)`` where ``metadata`` is a
        ``dict[str, list]`` ready to be passed to
        :class:`~concept.dataset.MultiSpeciesTokenizedDataset`.  Currently
        contains ``"species"``, whose value per file is taken from the
        ``species`` field of the split-config dict (``None`` for plain paths).
    """

    expanded: list = []
    species_list: list[str | None] = []
    for item in split_list:
        source_path = None
        effective_key = key
        current_species: str | None = None

        if isinstance(item, (dict, DictConfig)):
            # Wrapper syntax: {source: <split_config>, key: "train"|"val"}
            if "source" in item:
                effective_key = item.get("key", key)
                item = item["source"]

            # Capture species from the split config dict before extracting the file list.
            current_species = item.get("species", None)

            if "source_path" in item:
                source_path = item["source_path"]
            if effective_key is not None and effective_key in item:
                item = item[effective_key]

        if isinstance(item, str):
            item = [item]
        if source_path:
            item = [os.path.join(source_path, file) for file in item]
        expanded.extend(item)
        species_list.extend([current_species] * len(item))

    return expanded, {"species": species_list}


def load_pretrained_vocabulary(
    pretrained_vocabulary_dir: str,
    tokenizer: MultiSpeciesTokenizer,
) -> dict[str, torch.Tensor]:
    """Load pretrained gene embeddings and build a per-species vocabulary tensor.

    All CSV files in ``pretrained_vocabulary_dir`` are merged into a single
    gene-name → embedding mapping.  For each species registered in
    ``tokenizer``, a ``torch.Tensor`` of shape ``(vocab_size, pretrained_dim)``
    is produced and returned under the corresponding species key.

    Args:
        pretrained_vocabulary_dir: Directory containing ``*.csv`` files where
            the index column is gene name and the remaining columns are the
            embedding dimensions.
        tokenizer: A :class:`~concept.dataset.MultiSpeciesTokenizer` covering
            all species to embed.

    Returns:
        Dict mapping each species name to its pretrained embedding tensor.
    """
    import pandas as pd
    import glob

    csv_files = glob.glob(os.path.join(pretrained_vocabulary_dir, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {pretrained_vocabulary_dir}")

    logger.info(f"Loading {len(csv_files)} CSV files from {pretrained_vocabulary_dir}")

    pretrained_dict: dict[str, object] = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, index_col=0)
        for idx, row in df.iterrows():
            pretrained_dict[str(idx)] = row.values

    pretrained_dim = next(iter(pretrained_dict.values())).shape[0]
    logger.info(f"Loaded {len(pretrained_dict)} gene embeddings from {len(csv_files)} files")

    pretrained_vocabularies: dict[str, torch.Tensor] = {}
    for species in tokenizer.species:
        sp_tok = tokenizer.get_tokenizer(species)
        gene_names = sp_tok.decode(list(range(len(sp_tok.gene_mapping))))
        vocab_size = len(gene_names)
        pretrained_vocabulary = torch.zeros(vocab_size, pretrained_dim, dtype=torch.float)
        not_found: list[str] = []

        for idx, gene_name in enumerate(gene_names):
            if gene_name == tokenizer.CLS_VOCAB or gene_name == tokenizer.PAD_VOCAB:
                continue
            elif gene_name in pretrained_dict:
                pretrained_vocabulary[idx] = torch.FloatTensor(pretrained_dict[gene_name])
            else:
                not_found.append(gene_name)

        if not_found:
            logger.warning(
                f"[{species}] Pretrained embeddings not found for {len(not_found)} / {vocab_size} genes"
            )
        else:
            logger.info(f"[{species}] Pretrained embeddings found for all {vocab_size} genes")

        pretrained_vocabularies[species] = pretrained_vocabulary

    return pretrained_vocabularies


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
    val_check_interval = cfg.model.training.val_check_interval
    cfg.model.training.val_check_interval = (
        float(val_check_interval) if val_check_interval <= 1 else int(val_check_interval)
    )
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
            save_top_k=1,
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


from lightning.fabric.plugins.environments.slurm import SLURMEnvironment, _is_slurm_interactive_mode
from typing_extensions import override

class SLURMEnv(SLURMEnvironment):
    @staticmethod
    def _validate_srun_variables() -> None:
        """Checks for conflicting or incorrectly set variables set through `srun` and raises a useful error message.

        Right now, we only check for the most common user errors. See
        `the srun docs <https://slurm.schedmd.com/srun.html>`_
        for a complete list of supported srun variables.

        """
        # ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
        # if ntasks > 1 and "SLURM_NTASKS_PER_NODE" not in os.environ:
        #     raise RuntimeError(
        #         f"You set `--ntasks={ntasks}` in your SLURM bash script, but this variable is not supported."
        #         f" HINT: Use `--ntasks-per-node={ntasks}` instead."
        #     )

    @override
    def validate_settings(self, num_devices: int, num_nodes: int) -> None:
        if _is_slurm_interactive_mode():
            return
        # ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
        # if ntasks_per_node is not None and int(ntasks_per_node) != num_devices:
        #     raise ValueError(
        #         f"You set `devices={num_devices}` in Lightning, but the number of tasks per node configured in SLURM"
        #         f" `--ntasks-per-node={ntasks_per_node}` does not match. HINT: Set `devices={ntasks_per_node}`."
        #     )
        nnodes = os.environ.get("SLURM_NNODES")
        if nnodes is not None and int(nnodes) != num_nodes:
            raise ValueError(
                f"You set `num_nodes={num_nodes}` in Lightning, but the number of nodes configured in SLURM"
                f" `--nodes={nnodes}` does not match. HINT: Set `num_nodes={nnodes}`."
            )