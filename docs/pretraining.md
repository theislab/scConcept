# Pre-training from scratch

Use `src/concept/train.py` for large-scale distributed pre-training from
scratch. The `scConcept.train()` Python API is intended for light adaptation of
pretrained models or small experiments.

Before running large-scale pre-training, make sure the training files listed in
the split configuration files are available on disk.

## Environment setup

Clone the repository and create the virtual environment with the setup script:

```bash
git clone https://github.com/theislab/scConcept.git
cd scConcept
bash scripts/setup_env.sh
source .venv/bin/activate
```

The script uses the [uv](https://docs.astral.sh/uv/) package manager. If uv is
not installed, the script installs it and exits; restart your shell and run the
script again. It creates a Python 3.12 environment in `.venv`, installs all
dependencies, and builds `flash-attn`, so run it on a machine with a CUDA
toolchain (e.g. a GPU node).

## LaminDB setup

Follow the LaminDB setup guide before running the training pipeline:
<https://docs.lamin.ai/setup>.

## Expected project layout

The root training configuration is `src/concept/conf/config.yaml`. Set
`PATH.PROJECT_PATH` to a project directory with the following structure:

```text
PROJECT_PATH/
|-- data/
|   `-- <dataset_name>/
|       `-- h5ads/
|           |-- <adata_1>.h5ad
|           |-- <adata_2>.h5ad
|           `-- ...
|-- model_checkpoints/
|-- panels/
|   |-- hsapiens/
|   |   |-- panel_1.csv
|   |   |-- panel_2.csv
|   |   `-- ...
|   `-- mmusculus/
|       |-- panel_1.csv
|       |-- panel_2.csv
|       `-- ...
`-- references/
    |-- embeddings/
    |   `-- <embedding_name>/
    |       |-- <human_gene_embeddings>.csv
    |       |-- <mouse_gene_embeddings>.csv
    |       `-- ...
    `-- vocabulary/
        `-- token_mappings_per_specie/
            |-- hsapiens.csv
            |-- mmusculus.csv
            `-- ...
```

Panel CSV files are expected to contain an `Ensembl_ID` column. 
The per-species token mapping CSV files are expected to contain `gene_id` and 
`token` columns.

You don't have to build these reference files yourself. The per-species gene
token mapping files, the gene panels, and the ESM2 gene embeddings used to
pretrain scConcept are available on Hugging Face:

- Multi-species training:
  <https://huggingface.co/theislab/scConcept/tree/main/corpus360M%5Bmulti-species%5D-model170M>
- Human-only training:
  <https://huggingface.co/theislab/scConcept/tree/main/corpus40M-model30M-v3m>

Download them and place them under `PROJECT_PATH/references/` and
`PROJECT_PATH/panels/` following the layout above.

The gene embeddings directory is optional. Provide it only if you want to
initialize the model with pretrained gene embeddings. In that case, set
`PATH.PRETRAINED_VOCABULARY` to a directory that contains one CSV file per
species, for example:

```text
references/
`-- embeddings/
    `-- esm2_t30/
        |-- hsapiens.csv
        |-- mmusculus.csv
        `-- ...
```

The token mapping files look like:

```text
#hsapiens.csv
gene_id,gene_name,token
<cls>,,0
<pad>,,1
ENSG00000000003,TSPAN6,2
ENSG00000000005,TNMD,3
...
```

## Command-line overrides

The training script uses Hydra/OmegaConf. Any configuration value can be
overridden from the command line with dotted keys:

```bash
python src/concept/train.py PATH.PROJECT_PATH=/path/to/project model.training.max_steps=100000
```

## Configuration files

The training configuration is composed from Hydra YAML files in
`src/concept/conf/`.

`config.yaml`
: The root configuration. It selects the default model and datamodule configs,
  defines `PATH.*` locations, controls Weights & Biases logging, and stores
  resume/profiler settings. The most important required value is
  `PATH.PROJECT_PATH`, because the default data, panel, reference, and checkpoint
  paths are derived from it.

`model/ContrastiveModel.yaml`
: Model architecture and optimizer/training settings. This file controls the
  embedding size, transformer depth, attention heads, dropout, loss weights,
  Flash Attention flag, learning rate, scheduler, number of steps, GPU/device
  settings, validation interval, gradient accumulation, and checkpoint cadence.

`datamodule/DataModuleBasic.yaml`
: Default human-only datamodule with simple validations. It defines the
  species list, observation columns read from `adata.obs`, normalization,
  sampling strategy, train/validation splits, panel sampling settings, maximum
  token count, batch size, workers, and validation loaders.

`datamodule/split_*/split_*.yaml`
: Dataset split definitions. Each split file declares a `source_name`, a
  `species`, a `source_path`, and lists of `.h5ad` files for `train` and/or
  `val`. The training script expands these entries into concrete file paths and
  passes the species metadata to the multi-species tokenizer. A split entry can
  also choose a specific section with:

To use a different config group, override it on the command line. For example:

```bash
python src/concept/train.py \
  PATH.PROJECT_PATH=/path/to/project \
  datamodule=DataModuleAdvanced_2025_Primary \
  model.training.max_steps=1000000
```

## Common training options

Training/validation data
: The `datamodule` config group selects the training and validation data. For
  example, `datamodule=DataModuleAdvanced_2025_Primary` trains on the whole
  CELLxGENE 2025 census (primary cells).

Weights & Biases logging
: Set `wandb.entity` and `wandb.project` to your own W&B account and project,
  and pass a distinct `wandb.run_name` for each run. Logging is off by default;
  turn it on with `wandb.enabled=True`.

Fast local data copy
: Pass `PATH.LOCAL_DIR=/path/to/local/storage` to copy the training data to a
  fast local directory (e.g. each compute node's local scratch or SSD) before
  training starts. Reading from node-local storage is usually faster than from
  a shared network filesystem and speeds up training. Files already present
  and unchanged in `PATH.LOCAL_DIR` are not copied again.

  Note: the data is copied once per node only if the `PATH.LOCAL_DIR` path
  contains the string `localscratch`. Otherwise only the global rank-0 process
  copies it, which is enough when `PATH.LOCAL_DIR` is on storage shared by all
  nodes but not for multi-node training on node-local disks.

Spatial panel subsetting
: By default, training mixes in gene subsetting based on spatial gene panels
  to improve generalization to spatial data with limited gene panels. If that
  is not your main focus, disable it with
  `datamodule.dataset.train.panel_selection=random`.

Context length and dynamic batch sizing
: `datamodule.dataset.train.max_tokens` sets the maximum input context window,
  excluding zero-expressed genes (default `1000`). You can increase it
  substantially, e.g. to `20000` for the full transcriptome, but then also set
  `datamodule.dataset.train.max_total_seq_length=400_000` to enable dynamic
  batch sizing and avoid running out of GPU memory.

Example:

```bash
python src/concept/train.py \
  PATH.PROJECT_PATH=/path/to/project \
  PATH.LOCAL_DIR=/path/to/local/storage \
  datamodule=DataModuleAdvanced_2025_Primary \
  datamodule.dataset.train.panel_selection=random \
  datamodule.dataset.train.max_tokens=20000 \
  datamodule.dataset.train.max_total_seq_length=400_000 \
  wandb.enabled=True \
  wandb.entity=<your-entity> \
  wandb.project=<your-project> \
  wandb.run_name=<run-name>
```

## Launching a multi-GPU SLURM job

Submit the training command as a multi-GPU SLURM job, adjusting the example
above to your setup. Lightning reads the SLURM environment variables for
distributed training: by default it uses all GPUs on each node
(`model.training.devices=-1`) and takes the number of nodes from the job
allocation. Request one task per GPU. A minimal job script:

```bash
#!/bin/bash
#SBATCH --job-name=scconcept
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=48:00:00

cd /path/to/scConcept
source .venv/bin/activate

srun python src/concept/train.py \
  PATH.PROJECT_PATH=/path/to/project \
  datamodule=DataModuleAdvanced_2025_Primary \
  wandb.enabled=True \
  wandb.entity=<your-entity> \
  wandb.project=<your-project> \
  wandb.run_name=<run-name>
```

Add the other overrides from the example above as needed. Partition, account,
and resource flags depend on your cluster.

Checkpoints and the resolved training configuration are written to
`PATH.CHECKPOINT_ROOT`, which defaults to
`PATH.PROJECT_PATH/model_checkpoints`.
