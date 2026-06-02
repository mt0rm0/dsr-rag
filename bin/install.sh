#!/usr/bin/env bash
# Auto-installer for the DSR RAG workshop
# Requires: bash, curl
# Everything else (uv, Ollama, dependencies) is installed by this script.

set -euo pipefail

echo "==> Checking uv..."
if ! command -v uv &>/dev/null; then
    echo "    uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "    uv $(uv --version) ready."

echo ""
echo "==> Creating virtual environment and installing dependencies..."
uv sync
echo "    Dependencies installed."

echo ""
echo "==> Checking Ollama..."
if ! command -v ollama &>/dev/null; then
    echo "    Ollama not found — installing..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "    Ollama already installed."
fi

echo ""
echo "==> Pulling default LLM (llama3.2)..."
ollama pull llama3.2

echo ""
echo "============================================================"
echo " Setup complete!"
echo ""
echo " Verify your environment by running:"
echo "   uv run jupyter notebook notebooks/00_setup.ipynb"
echo ""
echo " To install bonus module dependencies:"
echo "   uv sync --extra bonus"
echo "============================================================"
