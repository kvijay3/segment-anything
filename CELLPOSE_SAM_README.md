# Cellpose-SAM: Recreating the Paper

This README provides comprehensive instructions for recreating the Cellpose-SAM paper: "Cellpose-SAM: superhuman generalization for cellular segmentation" by Pachitariu et al.

## Overview

Cellpose-SAM combines the pretrained transformer backbone of the Segment Anything Model (SAM) with the Cellpose framework to achieve superhuman performance in cellular segmentation. The key innovation is using SAM's image encoder while replacing its decoder with Cellpose's flow field prediction mechanism.

## Architecture Overview

### Original SAM Architecture
- **Image Encoder**: ViT-L transformer (305M out of 312M parameters)
- **Input Size**: 1024x1024 images with 16x16 patches
- **Decoder**: Complex mask and prompt decoders for sequential mask prediction

### Cellpose-SAM Modifications
- **Image Encoder**: Modified ViT-L transformer from SAM
- **Input Size**: Reduced to 256x256 images with 8x8 patches
- **Decoder**: Replaced with Cellpose flow field prediction
- **Attention**: Reverted to global attention on all layers (instead of local attention)

## Key Implementation Changes

### 1. Model Architecture Modifications

```python
# Key changes to implement:
# 1. Reduce input size from 1024x1024 to 256x256
# 2. Change patch size from 16x16 to 8x8
# 3. Adapt position embeddings via subsampling
# 4. Replace SAM decoder with Cellpose flow prediction
# 5. Use global attention instead of local attention
```

### 2. Training Dataset

The paper uses a combined dataset of **22,826 images** with **3,341,254 manual ROIs** from 18 publicly available datasets:

#### Core Datasets:
- **Cellpose (cyto2)**: 796 training images (59% sampling probability)
- **Cellpose Nuclei**: 1,025 training images (20% sampling probability)
- **TissueNet**: 2,601 training images (8% sampling probability)
- **LiveCell**: 3,188 training images (5% sampling probability)
- **Omnipose**: 392 training images (1-2% sampling probability each)
- **YeaZ**: 245 training images (2% sampling probability)
- **DeepBacs**: 155 training images (2% sampling probability)

#### Additional Datasets:
- Neurips 2022 challenge (504 selected images)
- MoNuSeg (37 training images)
- MoNuSAC (209 training images)
- CryoNuSeg (30 images)
- NuInsSeg (665 images)
- BCCD (1,169 images)
- CPM 15+17 (47 images)
- TNBC (50 images)
- LynSec (616 images)
- IHC TMA (231 images)
- CoNIC (3,863 images)
- PanNuke (6,053 images)

## Implementation Steps

### Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/MouseLand/cellpose.git
cd cellpose

# Install dependencies
pip install torch torchvision
pip install cellpose
pip install segment-anything

# Install additional requirements
pip install opencv-python scipy numpy matplotlib
pip install imagecodecs tifffile fastremap tqdm
pip install PyQt5 pyqtgraph
```

### Step 2: Download SAM Pretrained Weights

```bash
# Download SAM ViT-L model
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

### Step 3: Modify SAM Architecture

Create a new file `cellpose_sam_model.py`:

```python
import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from cellpose.models import CellposeModel

class CellposeSAM(nn.Module):
    def __init__(self, sam_checkpoint_path):
        super().__init__()
        
        # Load SAM model
        sam = sam_model_registry["vit_l"](checkpoint=sam_checkpoint_path)
        
        # Use only the image encoder from SAM
        self.image_encoder = sam.image_encoder
        
        # Modify for 256x256 input and 8x8 patches
        self.modify_encoder_for_cellpose()
        
        # Add Cellpose-style decoder for flow prediction
        self.flow_decoder = self.build_flow_decoder()
        
    def modify_encoder_for_cellpose(self):
        # Modify patch embedding from 16x16 to 8x8
        # Adapt position embeddings
        # Change to global attention on all layers
        pass
        
    def build_flow_decoder(self):
        # Build decoder that outputs 3 channels:
        # - Cell probability
        # - Horizontal flow
        # - Vertical flow
        return nn.ConvTranspose2d(1024, 3, kernel_size=8, stride=8)
        
    def forward(self, x):
        # Encode image
        features = self.image_encoder(x)
        
        # Decode to flow fields
        flows = self.flow_decoder(features)
        
        return flows
```

### Step 4: Training Configuration

```python
# Training hyperparameters from the paper
LEARNING_RATE = 5e-5
BATCH_SIZE = 256  # Distributed across 8 H200 GPUs
EPOCHS = 2000
WEIGHT_DECAY = 0.1
LAYER_DROP_RATE = 0.4

# Learning rate schedule
# - Linear warmup for first 10 epochs
# - Decrease by factor of 10 for last 100 epochs
# - Another factor of 10 for last 50 epochs
```

### Step 5: Data Augmentation Strategy

The paper uses extensive augmentation:

```python
# Augmentation pipeline:
# 1. Random rotation and flipping
# 2. Random resizing (scale factor 0.25-4x)
# 3. Random cropping to 256x256
# 4. Grayscale conversion (10% of time)
# 5. Contrast inversion (25% of time)
# 6. Channel permutation (for channel invariance)
# 7. Brightness/contrast jittering

# Image degradations (50% of batch):
# - Poisson noise
# - Gaussian blurring  
# - Downsampling
# - Anisotropic blurring
```

### Step 6: Loss Function

```python
def cellpose_loss(pred_flows, true_flows, pred_prob, true_prob):
    """
    Cellpose loss function:
    - MSE loss for XY flows (scaled by factor of 5)
    - Binary cross-entropy for cell probability
    """
    flow_loss = 5 * torch.mean((pred_flows - true_flows) ** 2)
    prob_loss = torch.nn.functional.binary_cross_entropy(pred_prob, true_prob)
    return flow_loss + prob_loss
```

## Dataset Preparation for Your Images

### Required Data Format

For your images and masks, you need to convert them to the Cellpose training format:

```python
import numpy as np
from cellpose import utils

def prepare_training_data(image_paths, mask_paths):
    """
    Convert your images and masks to Cellpose format
    
    Args:
        image_paths: List of paths to your images
        mask_paths: List of paths to corresponding masks
    """
    
    train_data = []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        # Load image (should be RGB or grayscale)
        img = utils.imread(img_path)
        
        # Load mask (should be integer labels, 0=background)
        mask = utils.imread(mask_path)
        
        # Ensure mask has unique integer labels for each cell
        if mask.max() == 1:  # Binary mask
            mask = utils.fill_holes_and_remove_small_masks(mask)
        
        # Compute flow fields from masks
        flows = utils.masks_to_flows(mask)
        
        train_data.append({
            'image': img,
            'masks': mask,
            'flows': flows
        })
    
    return train_data
```

### Data Requirements

1. **Images**: 
   - Format: PNG, TIFF, or JPG
   - Channels: 1-3 channels (grayscale, RGB, or fluorescence)
   - Size: Any size (will be resized during training)

2. **Masks**:
   - Format: PNG or TIFF
   - Values: Integer labels (0=background, 1,2,3...=individual cells)
   - Each cell should have a unique integer ID

### Dataset Structure

```
your_dataset/
├── images/
│   ├── img001.png
│   ├── img002.png
│   └── ...
├── masks/
│   ├── img001_mask.png
│   ├── img002_mask.png
│   └── ...
└── train_list.txt  # List of image-mask pairs
```

## Training Script

```python
import torch
from torch.utils.data import DataLoader
from cellpose_sam_model import CellposeSAM

def train_cellpose_sam():
    # Initialize model
    model = CellposeSAM("sam_vit_l_0b3195.pth")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=5e-5, 
        weight_decay=0.1
    )
    
    # Training loop
    for epoch in range(2000):
        for batch in dataloader:
            images, flows, probs = batch
            
            # Forward pass
            pred_flows = model(images)
            
            # Compute loss
            loss = cellpose_loss(pred_flows, flows, pred_flows[:, 0], probs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## Key Performance Metrics

The paper evaluates using:

1. **Error Rate**: `(FN + FP) / (TP + FN)`
2. **Average Precision (AP)**: `TP / (TP + FP + FN)`
3. **IoU Threshold**: 0.5 for matching predictions to ground truth

### Expected Results

- **Cellpose-SAM**: Error rate ~0.163 (approaching human consensus)
- **Previous models**: Error rates ~0.29-0.33 (inter-annotator level)
- **Human consensus estimate**: ~0.128 error rate

## Hardware Requirements

### Training
- **Recommended**: 8x H200 GPUs (as used in paper)
- **Minimum**: Single GPU with 8-12GB VRAM (batch size 1)
- **Training time**: ~20 hours on 8x H200 GPUs

### Inference
- **GPU**: Any modern GPU (RTX 4060 or better)
- **Runtime**: ~0.4 seconds per 256x256 image

## Next Steps

1. **Set up environment** and install dependencies
2. **Download and prepare datasets** using the dataset list above
3. **Implement the model architecture** modifications
4. **Prepare your training data** in the correct format
5. **Start training** with the provided hyperparameters
6. **Evaluate performance** using the metrics from the paper

## Additional Resources

- [Original Cellpose Paper](https://www.nature.com/articles/s41592-020-01018-x)
- [SAM Paper](https://arxiv.org/abs/2304.02643)
- [Cellpose GitHub](https://github.com/MouseLand/cellpose)
- [SAM GitHub](https://github.com/facebookresearch/segment-anything)

## Questions?

Feel free to ask about any specific implementation details or if you need help with dataset preparation!

