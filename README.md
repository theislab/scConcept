# scConcept

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/scConcept/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/scConcept

This repository contains the python package to train and use scConcept (Single-cell contrastive cell pre-training) method for single-cell transcriptomics.

<!-- ## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][]. -->

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

### Default installation

Install the latest release of `sc-concept` from [PyPI][]:

```bash
pip install sc-concept
```

### Latest development version

To install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/theislab/scConcept.git@main
```

### Training from scratch with Flash Attention

The standard installation is enough for loading pretrained models, extracting embeddings, and light adaptation. If you want to train scConcept from scratch with [Flash Attention][], use one of the following options.

1. Recommended: `cd` to the project root and run [`./scripts/setup_env.sh`](https://github.com/theislab/scConcept/blob/main/scripts/setup_env.sh), which installs uv if needed and creates a virtual environment with the training dependencies.

2. Manual: make sure a CUDA-enabled version of PyTorch is installed. More information is available in the [PyTorch installation guide](https://pytorch.org/get-started/locally/). Then install Flash Attention:

```bash
MAX_JOBS=4 pip install "flash-attn>=2.7" --no-build-isolation
```

This can take up to an hour depending on the system specifications and whether a pre-built release of `flash-attn` is available for your exact versions of Python, PyTorch, and CUDA. If this takes long, we recommend using the setup script instead.


## How to use

scConcept provides a simple API to load and adapt [pre-trained models](https://huggingface.co/theislab/scConcept/tree/main) and extract embeddings from scRNA-seq data. Here's a basic example:

```python
from concept import scConcept
import scanpy as sc

# Load your single-cell data
adata = sc.read_h5ad("your_data.h5ad")

# Initialize scConcept and load a pretrained model
concept = scConcept(cache_dir='./cache/')

# Option 1: Load a model directly from HuggingFace
concept.load_config_and_model(model_name='corpus40M-model30M') 

# Option 2: Load any local model
concept.load_config_and_model(
    config='<path-to-config.yaml>',
    model_path='<path-to-model.ckpt>',
    gene_mappings_path='<path-to-gene-mapping.pkl>',
)

# Extract embeddings --> adata.var['gene_id']: ENSGXXXXXXXXXXX
result = concept.extract_embeddings(adata=adata, gene_id_column='gene_id')

# Use embeddings for downstream analysis
adata.obsm['X_scConcept'] = result['cls_cell_emb']

# Adapt a pre-trained model on your own data
concept.train(adata, max_steps=10000, batch_size=128) 

# Important: For multiple datasets pass them separately
concept.train([adata1, adata2, ...], max_steps=20000, batch_size=128) 

result = concept.extract_embeddings(adata=adata, gene_id_column='gene_id')
adata.obsm['X_scConcept_adapted'] = result['cls_cell_emb']
```
<!-- For more detailed example, see the [notebook example](docs/notebooks/embedding_extraction.ipynb). -->


## Large-scale pre-training from scratch

`scConcept.train()` is only for light adaptation of pretrained models or small trainings on the fly. Use [train.py](https://github.com/theislab/scConcept/blob/main/src/concept/train.py) for distributed model pre-training from scratch over large corpus of data.

Before using `train.py` follow the instructions on [lamindb](https://github.com/laminlabs/lamindb) for setting up a lamin instance.


## Troubleshooting

If you encounter an error when loading a pre-trained model, try the following:

1. Remove the repository and clone the most recent version
2. Remove the cache directory (`cache/` by default)
3. Run again

This will force a fresh download of the pre-trained model and should resolve most loading issues.

<!-- ## Release notes

See the [changelog][]. -->

<!-- ## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][]. -->

## Citation

> Bahrami, M., Tejada-Lapuerta, A., Becker, S., Hashemi G, F.S. and Theis, F.J., 2025. scConcept: Contrastive pretraining for technology-agnostic single-cell representations beyond reconstruction. bioRxiv, pp.2025-10. doi: https://doi.org/10.1101/2025.10.14.682419

[uv]: https://github.com/astral-sh/uv
[Flash Attention]: https://github.com/Dao-AILab/flash-attention
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/theislab/scConcept/issues
[tests]: https://github.com/theislab/scConcept/actions/workflows/test.yaml
[documentation]: https://scConcept.readthedocs.io
[changelog]: https://scConcept.readthedocs.io/en/latest/changelog.html
[api documentation]: https://scConcept.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/scConcept
