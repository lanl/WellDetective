#!/bin/bash

# WellDetective Conda Environment Setup Script for macOS/Linux
# This script creates a conda environment and installs dependencies

set -e  # Exit on error

# Determine conda installation location
# Check common installation locations for conda
CONDA_SH=""

# Check for Anaconda3 in home directory
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
# Check for Miniconda3 in home directory
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
# Check for Anaconda3 in opt (common brew location)
elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="/opt/anaconda3/etc/profile.d/conda.sh"
# Check for Miniconda3 in opt
elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="/opt/miniconda3/etc/profile.d/conda.sh"
# Check for conda in homebrew location (Apple Silicon)
elif [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
    CONDA_SH="/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
# Check for conda in homebrew location (Intel Mac)
elif [ -f "/usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
    CONDA_SH="/usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh"
else
    echo "Error: Could not find Anaconda or Miniconda installation."
    echo ""
    echo "Expected conda.sh in one of the following locations:"
    echo "  $HOME/anaconda3/etc/profile.d/conda.sh"
    echo "  $HOME/miniconda3/etc/profile.d/conda.sh"
    echo "  /opt/anaconda3/etc/profile.d/conda.sh"
    echo "  /opt/miniconda3/etc/profile.d/conda.sh"
    echo "  /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    echo "  /usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    echo ""
    echo "Please install Miniconda or Anaconda, or run:"
    echo "  which conda"
    echo "to find your conda installation."
    exit 1
fi

echo "Found conda at: $CONDA_SH"

# Source conda.sh to make conda command available
source "$CONDA_SH"

# Change to src directory
cd "./src" || {
    echo "Error: Could not find ./src directory"
    exit 1
}

echo "Creating conda environment: WellDetectiveEnv"
echo "Note: Using Python 3.11 for better package compatibility"
conda create -n WellDetectiveEnv python=3.11 -y

echo "Activating conda environment"
conda activate WellDetectiveEnv

echo "Installing all packages via conda-forge (pre-built binaries, no compilation needed)"
conda install -c conda-forge numpy pandas scipy scikit-learn xarray matplotlib pyproj jupyter notebook numba llvmlite harmonica xrft simplekml -y

echo "Installing pyIGRF14 via pip (not available in conda-forge)"
pip install pyIGRF14==1.0.4

echo "Installing WellDetective package in editable mode"
pip install -e ./

echo ""
echo "================================="
echo "Setup complete!"
echo "================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate WellDetectiveEnv"
echo ""
echo "To deactivate when done, run:"
echo "  conda deactivate"
echo ""
