#!/usr/bin/env bash
# ARM installation script for Raspberry Pi and other ARM processors.
# Sets up Miniforge/Conda, MPI, and Python dependencies for the star counter project.

set -euo pipefail

ENV_NAME="astro-mpi"
PYTHON_VERSION="3.11"

if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "armv7l" ]]; then
  echo "This script is intended for ARM-based systems (Raspberry Pi / ARM64)."
  echo "Detected architecture: $(uname -m)"
  exit 1
fi

echo "==> Updating package lists"
sudo apt update

echo "==> Installing system dependencies"
sudo apt install -y wget build-essential libopenmpi-dev openmpi-bin git

if ! command -v conda >/dev/null 2>&1; then
  echo "==> Installing Miniforge for ARM"
  cd /tmp
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
  bash Miniforge3-Linux-aarch64.sh -b -p "$HOME/miniforge3"
  rm -f Miniforge3-Linux-aarch64.sh
  eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
else
  echo "==> Conda already installed"
  eval "$(conda shell.bash hook)"
fi

echo "==> Creating Conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
conda activate "$ENV_NAME"

echo "==> Installing Python packages"
conda install -y numpy
pip install opencv-python mpi4py

echo "==> Verifying installation"
python -c "import numpy, cv2, mpi4py; print('numpy', numpy.__version__); print('cv2', cv2.__version__); print('mpi4py', mpi4py.__version__)"
mpirun --version

echo "==> Installation complete. Activate with: conda activate $ENV_NAME"
