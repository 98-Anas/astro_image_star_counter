#!/usr/bin/env bash
# AMD/x86_64 installation script for desktop and server systems.
# Sets up Miniconda/Conda, MPI, and Python dependencies for the star counter project.

set -euo pipefail

ENV_NAME="astro-mpi"
PYTHON_VERSION="3.11"

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "This script is intended for AMD/x86_64 systems."
  echo "Detected architecture: $(uname -m)"
  exit 1
fi

echo "==> Updating package lists"
sudo apt update

echo "==> Installing system dependencies"
sudo apt install -y wget build-essential libopenmpi-dev openmpi-bin git

if ! command -v conda >/dev/null 2>&1; then
  echo "==> Installing Miniconda for x86_64"
  cd /tmp
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
  rm -f Miniconda3-latest-Linux-x86_64.sh
  eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
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
