# scConcept

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/mojtababahrami/scConcept/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/scConcept

This repository contains the python package to train and use scConcept (Single-cell contrastive cell pre-training) method for single-cell transcriptomics.

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install scConcept:

<!--
1) Install the latest release of `scConcept` from [PyPI][]:

```bash
pip install scConcept
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/mojtababahrami/scConcept.git@main
```

## How to use

scConcept provides a simple API to load pre-trained model, adapt and extract embeddings from scRNA-seq data. Here's a basic example:

```python
from concept.scConcept import scConcept
import scanpy as sc

# Load your single-cell data
adata = sc.read_h5ad("your_data.h5ad")

# Initialize scConcept and load a pretrained model
concept = scConcept(cache_dir='./cache/')
concept.load_config_and_model(model_name='Corpus-30M')

# Extract embeddings
# --> adata.var['gene_id']: ENSGXXXXXXXXXXX
result = concept.extract_embeddings(adata=adata, gene_id_column='gene_id')

# Use embeddings for downstream analysis
adata.obsm['X_scConcept'] = result['cls_cell_emb']
```

For more detailed examples and tutorials, see the [documentation][] and the [notebook examples](docs/notebooks/embedding_extraction.ipynb).

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> Bahrami, M., Tejada-Lapuerta, A., Becker, S., Hashemi G, F.S. and Theis, F.J., 2025. scConcept: Contrastive pretraining for technology-agnostic single-cell representations beyond reconstruction. bioRxiv, pp.2025-10. doi: https://doi.org/10.1101/2025.10.14.682419

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/mojtababahrami/scConcept/issues
[tests]: https://github.com/mojtababahrami/scConcept/actions/workflows/test.yaml
[documentation]: https://scConcept.readthedocs.io
[changelog]: https://scConcept.readthedocs.io/en/latest/changelog.html
[api documentation]: https://scConcept.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/scConcept
