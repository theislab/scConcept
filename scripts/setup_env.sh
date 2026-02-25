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

# A flash-attn build is available for this Python version
echo "📦 Creating virtual environment..."
uv venv --python 3.12

echo "📦 Installing flash-attn..."
MAX_JOBS=4 uv pip install "flash-attn==2.8.3" --no-build-isolation

echo "🔧 Activating virtual environment..."
source .venv/bin/activate

echo "📥 Installing dependencies..."
uv sync

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