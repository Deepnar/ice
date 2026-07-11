#!/usr/bin/env bash
set -e

echo "===== ICE Installer ====="

# Prerequisites (assumes Arch/CachyOS)
echo "Installing system packages..."
sudo pacman -S --needed docker docker-compose postgresql python pyenv cmake

# Python environment
pyenv install 3.11.9 --skip-existing
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate

# Python dependencies
pip install uv
uv sync

# Docker services
docker compose -f docker/docker-compose.yml up -d

# Database
echo "Running migrations..."
uv run alembic upgrade head

# Background model (C7 shared-first): the default shared mode reuses the main
# Ollama model — nothing to pull. Dedicated mode only:
#   huggingface-cli download Qwen/Qwen2.5-3B-Instruct-AWQ

echo "===== ICE installed! Start with: ./ice ====="