scConcept
=======

This repository contains the python package to train and use scConcept (Single-cell contrastive cell pre-training) method for single-cell transcriptomics data.

Installation
------------

## Option 1: Using uv (Recommended)

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a virtual environment and install dependencies:
```bash
sh ./scripts/setup_uv.sh
```

## Option 2: Using pip

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install the package and dependencies:
```bash
pip install -e .
```

3. Install [Flash Attention 2](https://github.com/Dao-AILab/flash-attention):
```bash
pip install flash-attn==2.7.* --no-build-isolation
```

4. Install `lamin-dataloader` (optional: only required for training over large number of anndata objects):
```bash
pip install git+https://github.com/theislab/lamin_dataloader.git
```


<!-- Usage
------------ -->


Licence
-------
`scConcept` is licensed under the [MIT License](https://opensource.org/licenses/MIT)
