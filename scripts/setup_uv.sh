#!/bin/bash

# Setup script for uv package manager migration

set -e

echo "🚀 Setting up contrastive-transformer with uv package manager..."

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
uv sync --dev

# Install flash-attn separately (needs torch to be installed first)
echo "⚡ Installing flash-attn..."
uv pip install flash-attn==2.7.* --no-build-isolation

# Install lamin-dataloader in editable mode
echo "🔗 Installing lamin-dataloader in editable mode..."
uv pip install -e /home/icb/mojtaba.bahrami/projects/lamin-dataloader

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