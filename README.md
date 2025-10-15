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

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the environment:
```bash
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

3. Install the package and dependencies:
```bash
pip install -e .
```


<!-- Usage
------------ -->


Licence
-------
`scConcept` is licensed under the [MIT License](https://opensource.org/licenses/MIT)
