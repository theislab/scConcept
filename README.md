scConcept
=======

Single-Cell Contrastive Transformer

Installation
------------

## Option 1: Using uv (Recommended)

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate     # On Windows
```

3. Install dependencies and the package in development mode:
```bash
uv sync --dev
uv pip install flash-attn==2.7.* --no-build-isolation
uv pip install -e /home/icb/mojtaba.bahrami/projects/lamin-dataloader
uv pip install -e .
```

## Option 2: Using conda/mamba (Legacy)

1. Create the conda/mamba environment:
```bash
conda env create -f environment.yml
```

2. Activate the environment:
```bash
conda activate concept
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Development Setup

For development, install with dev dependencies:
```bash
uv sync --dev
uv pip install flash-attn ==2.7.* --no-build-isolation
uv pip install -e /home/icb/mojtaba.bahrami/projects/lamin-dataloader
uv pip install -e .
```

For Jupyter support:
```bash
uv sync --extra jupyter
```

## Dependencies

This project depends on a local development package `lamin-dataloader` located at:
`/home/icb/mojtaba.bahrami/projects/lamin-dataloader`

This package is automatically installed in editable mode during setup.

### Flash Attention

Flash Attention is installed separately because it requires PyTorch to be already installed during its build process. The installation uses the `--no-build-isolation` flag to handle this dependency requirement.

Usage
------------

1. To Train from scrath refer to the example script [train.sh](https://github.com/theislab/scConcept-reproducibility/blob/main/ct_rep/get_embs/train.sh) in the reproducibility repo

2. To get the embeddings from a pre-trained model refer to the example script [get_embs.sh](https://github.com/theislab/scConcept-reproducibility/blob/main/ct_rep/get_embs/get_embs.sh) in the reproducibility repo

3. To validate a pre-trained model on a new hold-out dataset refert the example script [validate.sh](https://github.com/theislab/scConcept-reproducibility/blob/main/scripts/validate.sh) in the reproducibility repo

4. To adapt a pre-trained model for a new hold-out dataset refer to the example script [adapt.sh](https://github.com/theislab/scConcept-reproducibility/blob/main/scripts/adapt.sh) in the reproducibility repo

Licence
-------
`scConcept` is licensed under the [MIT License](https://opensource.org/licenses/MIT)
