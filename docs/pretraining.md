# Pre-training from scratch

Use `src/concept/train.py` for large-scale distributed pre-training from
scratch. The `scConcept.train()` Python API is intended for light adaptation of
pretrained models or small experiments.

Before running large-scale pre-training, make sure the training files listed in
the split configuration files are available on disk.

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
`token` columns. Example panel CSV files and token mapping files are available in 
the scConcept model repository on Hugging Face: <https://huggingface.co/theislab/scConcept/tree/main>.

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
  datamodule=DataModuleAdvanced \
  model.training.max_steps=1000000
```

On SLURM, launch the same command through `srun` or your site-specific job
script. Lightning reads SLURM environment variables and uses them for
distributed training when available.

Checkpoints and the resolved training configuration are written to
`PATH.CHECKPOINT_ROOT`, which defaults to
`PATH.PROJECT_PATH/model_checkpoints`.
