import h5py
import filecmp
import logging
import os
import shutil
from datetime import timedelta
import torch
from concept.dataset import MultiSpeciesTokenizer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class _GlobalRankZeroLoggingFilter(logging.Filter):
    """Allow log records only from global rank zero."""

    def filter(self, record: logging.LogRecord) -> bool:
        return rank_zero_only.rank == 0


def setup_logging() -> None:
    """Configure package logging and restrict emission to global rank zero."""

    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    configure_global_rank_zero_logging()


def configure_global_rank_zero_logging() -> None:
    """Attach a filter so only global rank zero emits Python log records."""

    rank_zero_filter = _GlobalRankZeroLoggingFilter()
    root_logger = logging.getLogger()

    if not root_logger.handlers:
        root_logger.addFilter(rank_zero_filter)
        return

    for handler in root_logger.handlers:
        if any(isinstance(existing_filter, _GlobalRankZeroLoggingFilter) for existing_filter in handler.filters):
            continue
        handler.addFilter(rank_zero_filter)


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


def build_species_gene_mappings(
    gene_mappings_path: str,
    species: list[str],
) -> dict[str, dict]:
    """Build a ``{species: {gene_id: token_id}}`` mapping from per-species CSV files.

    Expects one CSV file per species at ``<gene_mappings_path>/<species>.csv``
    with at minimum a ``gene_id`` index column and a ``token`` column.

    Args:
        gene_mappings_path: Directory containing ``<species>.csv`` files.
        species: List of species names to load.

    Returns:
        Mapping suitable for passing to
        :class:`~concept.dataset.MultiSpeciesTokenizer`.
    """
    import pandas as pd

    return {
        sp: pd.read_csv(os.path.join(gene_mappings_path, f"{sp}.csv"), index_col="gene_id")["token"].to_dict()
        for sp in species
    }


def load_pretrained_vocabulary(
    pretrained_vocabulary_dir: str,
    tokenizer: MultiSpeciesTokenizer,
    pretrained_dim: int,
) -> dict[str, torch.Tensor]:
    """Load pretrained gene embeddings and build a per-species vocabulary tensor.

    For each species registered in ``tokenizer``, the file
    ``<pretrained_vocabulary_dir>/<species>.csv`` is loaded (index column is
    gene name, remaining columns are embedding dimensions).  If no file exists
    for a species the returned tensor is all zeros.

    Args:
        pretrained_vocabulary_dir: Directory containing ``<species>.csv`` files
            where the index column is gene name and the remaining columns are
            the embedding dimensions.
        tokenizer: A :class:`~concept.dataset.MultiSpeciesTokenizer` covering
            all species to embed.
        pretrained_dim: Dimensionality of the pretrained embeddings
            (``model.dim_pretrained_vocab`` in the config).

    Returns:
        Dict mapping each species name to its pretrained embedding tensor of
        shape ``(vocab_size, pretrained_dim)``.
    """
    import pandas as pd

    pretrained_vocabularies: dict[str, torch.Tensor] = {}

    for species in tokenizer.species:
        sp_tok = tokenizer.get_tokenizer(species)
        gene_names = sp_tok.decode(list(range(len(sp_tok.gene_mapping))))
        vocab_size = len(gene_names)

        csv_path = os.path.join(pretrained_vocabulary_dir, f"{species}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"[{species}] Pretrained vocabulary file not found: {csv_path}."
            )

        logger.info(f"[{species}] Loading pretrained embeddings from {csv_path}")
        df = pd.read_csv(csv_path, index_col=0)
        pretrained_dict: dict[str, object] = {str(idx): row.values for idx, row in df.iterrows()}

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


def infer_species(gene_ids: set[str], tokenizer: "MultiSpeciesTokenizer") -> str:
    """Infer the species of a set of gene IDs by overlap with tokenizer vocabularies.

    Args:
        gene_ids: Set of gene identifiers (e.g. Ensembl IDs) from the dataset.
        tokenizer: A :class:`~concept.dataset.MultiSpeciesTokenizer`.

    Returns:
        The species name with the highest unique overlap.

    Raises:
        ValueError: If no overlap is found, or if the overlap is tied between
            multiple species.
    """
    special_tokens = {tokenizer.CLS_VOCAB, tokenizer.PAD_VOCAB}
    overlaps = {
        sp: len(gene_ids & (set(tok.gene_mapping.keys()) - special_tokens))
        for sp, tok in tokenizer._tokenizers.items()
    }
    best_overlap = max(overlaps.values())
    if best_overlap == 0:
        raise ValueError(
            f"Cannot infer species: no overlap between the provided genes and any species vocabulary. "
            f"Overlaps: {overlaps}. Please specify the 'species' argument explicitly."
        )
    tied = [s for s, v in overlaps.items() if v == best_overlap]
    if len(tied) > 1:
        raise ValueError(
            f"Cannot infer species: ambiguous overlap between the provided genes and multiple species {tied}. "
            f"Overlaps: {overlaps}. Please specify the 'species' argument explicitly."
        )
    return tied[0]


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
    config_file = bash_cfg.initialize.get("config_file", None)
    if config_file is not None:
        logger.info("Loading config from local file %s ...", config_file)
        cfg = OmegaConf.load(config_file)
    else:
        import wandb

        wandb.login()
        api = wandb.Api()
        run = api.run(f"{bash_cfg.wandb.entity}/{bash_cfg.wandb.project}/{bash_cfg.initialize.run_id}")
        logger.info(f"Resuming training for {run.id} ...")
        cfg = DictConfig(run.config)

        if not cfg.initialize.create_new_run:
            os.environ["WANDB_TAGS"] = ",".join(run.tags) + "," + os.environ.get("WANDB_TAGS", "")

    cfg = OmegaConf.merge(cfg, bash_cfg)


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
