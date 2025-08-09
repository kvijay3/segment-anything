#!/bin/bash

# Setup script for Cellpose-SAM environment
# This script sets up the environment and downloads necessary models

echo "Setting up Cellpose-SAM environment..."

# Create conda environment (optional)
read -p "Do you want to create a new conda environment? (y/n): " create_env
if [ "$create_env" = "y" ]; then
    read -p "Enter environment name (default: cellpose-sam): " env_name
    env_name=${env_name:-cellpose-sam}
    
    echo "Creating conda environment: $env_name"
    conda create -n $env_name python=3.9 -y
    conda activate $env_name
fi

# Install PyTorch (adjust for your CUDA version)
echo "Installing PyTorch..."
read -p "Do you have CUDA GPU? (y/n): " has_cuda
if [ "$has_cuda" = "y" ]; then
    # Install CUDA version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    # Install CPU version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements.txt

# Create directories
echo "Creating directories..."
mkdir -p data/images
mkdir -p data/masks
mkdir -p checkpoints
mkdir -p results

# Download SAM model weights
echo "Downloading SAM model weights..."
if [ ! -f "sam_vit_l_0b3195.pth" ]; then
    echo "Downloading SAM ViT-L model (2.6GB)..."
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
    echo "SAM model downloaded successfully!"
else
    echo "SAM model already exists."
fi

# Download example datasets (optional)
read -p "Do you want to download example datasets? (y/n): " download_data
if [ "$download_data" = "y" ]; then
    echo "Downloading example datasets..."
    
    # Create data directory
    mkdir -p example_data
    cd example_data
    
    # Download some example datasets
    echo "Downloading Cellpose example data..."
    wget -O cellpose_example.zip "https://www.cellpose.org/static/data/cellpose_example_data.zip"
    unzip cellpose_example.zip
    
    echo "Example data downloaded to example_data/"
    cd ..
fi

# Test installation
echo "Testing installation..."
python -c "
import torch
import cv2
import numpy as np
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
print('OpenCV version:', cv2.__version__)
print('NumPy version:', np.__version__)

try:
    import cellpose
    print('Cellpose version:', cellpose.__version__)
except ImportError:
    print('Cellpose not installed')

try:
    import segment_anything
    print('Segment Anything installed successfully')
except ImportError:
    print('Segment Anything not installed')
"

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Place your training images in data/images/"
echo "2. Place your corresponding masks in data/masks/"
echo "3. Run training with:"
echo "   python train_cellpose_sam.py --image_folder data/images --mask_folder data/masks --sam_checkpoint sam_vit_l_0b3195.pth"
echo ""
echo "For more information, see CELLPOSE_SAM_README.md"

