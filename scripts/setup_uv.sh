#!/bin/bash

# Setup script for uv package manager migration

set -e

echo "🚀 Setting up concept with uv package manager..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✅ uv installed successfully!"
    echo "Please restart your shell or run: source ~/.bashrc"
    exit 1
fi

echo "✅ uv is already installed"

# Create virtual environment
echo "📦 Creating virtual environment..."
uv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
uv sync --all-extras

# Check for CUDA availability
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        echo "🔍 CUDA detected via nvidia-smi"
        return 0
    elif python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        echo "🔍 CUDA detected via PyTorch"
        return 0
    else
        echo "⚠️  CUDA not detected"
        return 1
    fi
}

# Install flash-attn only if CUDA is available
if check_cuda; then
    echo "⚡ Installing flash-attn (CUDA detected)..."
    MAX_JOBS=10 uv pip install flash-attn>=2.7 --no-build-isolation
else
    echo "⏭️  Skipping flash-attn installation (CUDA not available)"
fi

# Install lamin-dataloader from GitHub
echo "🔗 Installing lamin-dataloader from GitHub..."
uv pip install git+https://github.com/theislab/lamin_dataloader.git

# Install package in development mode
echo "🔗 Installing package in development mode..."
uv pip install -e .

echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To install additional dependencies:"
echo "  uv add <package-name>"
echo ""
echo "To update dependencies:"
echo "  uv sync --upgrade" 