# Installation

scConcept requires Python 3.12 or newer. Python 3.10 is not supported by the
current package metadata, so create a Python 3.12+ environment before installing
the package.

## Default installation

Install the latest release from PyPI:

```bash
pip install sc-concept
```

To install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/theislab/scConcept.git@main
```

## HPC installations

On HPC systems, installation failures often come from dependencies being built
from source instead of using pre-built binary packages. This can happen when the
system Python, compiler stack, or package resolver cannot find compatible
wheels.

Recommended HPC setup:

```bash
conda create -n scconcept -c conda-forge python=3.12
conda activate scconcept
conda install -c conda-forge numpy h5py pyarrow
pip install sc-concept
```

Installing `numpy`, `h5py`, and `pyarrow` from `conda-forge` first avoids common
source-build requirements that may be unavailable on shared clusters:

- NumPy source builds require a sufficiently recent compiler toolchain. Use GCC
  9.3 or newer if your cluster builds NumPy from source.
- Building `h5py` from source requires the HDF5 development libraries and a
  compatible compiler stack.
- Building `pyarrow` from source requires CMake 3.25 or newer and Arrow C++
  components, which are often not installed on typical HPC login or compute
  nodes.

If your HPC module system provides these packages, load the corresponding Python,
GCC, CMake, HDF5, and Arrow modules before installing. Otherwise, prefer the
pre-built `conda-forge` packages shown above.

## Optional Flash Attention speedup

The standard installation is enough for loading pretrained models, extracting
embeddings, and light adaptation. For faster inference, embedding extraction,
adaptation, or large-scale training, install Flash Attention with one of the
following options.

1. From the project root, run `./scripts/setup_env.sh`. The script creates a
   Python 3.12 environment with `uv`, installs the project dependencies, and then
   installs Flash Attention.

2. Install it manually after a CUDA-enabled PyTorch build is available:

```bash
MAX_JOBS=4 pip install "flash-attn>=2.7" --no-build-isolation
```

This can take up to an hour depending on the system specifications and whether a
pre-built `flash-attn` release is available for your exact Python, PyTorch, and
CUDA versions.
