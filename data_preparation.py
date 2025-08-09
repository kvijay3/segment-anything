"""
Data Preparation for Cellpose-SAM Training

This module provides utilities to prepare your image and mask data
for training the Cellpose-SAM model.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import cv2
from pathlib import Path
import json

try:
    from cellpose import utils, transforms
    from cellpose.io import imread
except ImportError:
    print("Warning: cellpose not installed. Please install with: pip install cellpose")
    # Fallback imports
    import imageio
    imread = imageio.imread

class CellposeSAMDataset(Dataset):
    """
    Dataset class for Cellpose-SAM training
    
    Handles loading images and masks, computing flow fields,
    and applying augmentations.
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 mask_paths: List[str],
                 image_size: int = 256,
                 augment: bool = True,
                 normalize: bool = True):
        """
        Initialize dataset
        
        Args:
            image_paths: List of paths to training images
            mask_paths: List of paths to corresponding masks
            image_size: Target image size (will resize to image_size x image_size)
            augment: Whether to apply data augmentation
            normalize: Whether to normalize images
        """
        assert len(image_paths) == len(mask_paths), "Number of images and masks must match"
        
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        
        print(f"Initialized dataset with {len(image_paths)} image-mask pairs")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample
        
        Returns:
            dict with keys: 'image', 'flows', 'cell_prob'
        """
        # Load image and mask
        image = self._load_image(self.image_paths[idx])
        mask = self._load_mask(self.mask_paths[idx])
        
        # Ensure same spatial dimensions
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Compute flow fields from mask
        flows, cell_prob = self._compute_flows(mask)
        
        # Apply augmentations
        if self.augment:
            image, flows, cell_prob = self._apply_augmentations(image, flows, cell_prob)
        
        # Resize to target size
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        flows = cv2.resize(flows, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        cell_prob = cv2.resize(cell_prob, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        
        # Normalize image
        if self.normalize:
            image = self._normalize_image(image)
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        flows = torch.from_numpy(flows).float()
        cell_prob = torch.from_numpy(cell_prob).float()
        
        # Ensure correct dimensions
        if len(image.shape) == 2:  # Grayscale
            image = image.unsqueeze(0)
        elif len(image.shape) == 3:  # RGB
            image = image.permute(2, 0, 1)
        
        # Combine flows and cell_prob into single tensor [3, H, W]
        flow_tensor = torch.stack([cell_prob, flows[..., 0], flows[..., 1]], dim=0)
        
        return {
            'image': image,
            'flows': flow_tensor
        }
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load and preprocess image"""
        try:
            image = imread(path)
        except:
            # Fallback to cv2
            image = cv2.imread(path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image is None:
            raise ValueError(f"Could not load image: {path}")
        
        # Ensure float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)
            if image.max() > 1.0:
                image = image / 255.0
        
        return image
    
    def _load_mask(self, path: str) -> np.ndarray:
        """Load mask with integer labels"""
        try:
            mask = imread(path)
        except:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            raise ValueError(f"Could not load mask: {path}")
        
        # Ensure integer type
        mask = mask.astype(np.int32)
        
        return mask
    
    def _compute_flows(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute flow fields from mask using Cellpose method
        
        Args:
            mask: Integer mask with cell labels
            
        Returns:
            flows: [H, W, 2] flow field (dY, dX)
            cell_prob: [H, W] cell probability
        """
        try:
            # Use Cellpose utils if available
            flows = utils.masks_to_flows(mask)
            flow_field = flows[0]  # [H, W, 2]
            cell_prob = (mask > 0).astype(np.float32)
        except:
            # Fallback implementation
            flow_field, cell_prob = self._compute_flows_fallback(mask)
        
        return flow_field, cell_prob
    
    def _compute_flows_fallback(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback flow computation without Cellpose
        
        This is a simplified version - the full Cellpose implementation
        is more sophisticated.
        """
        H, W = mask.shape
        flows = np.zeros((H, W, 2), dtype=np.float32)
        cell_prob = (mask > 0).astype(np.float32)
        
        # For each cell, compute flows pointing toward center
        unique_labels = np.unique(mask)
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            
            cell_mask = mask == label
            if not np.any(cell_mask):
                continue
            
            # Find center of mass
            y_coords, x_coords = np.where(cell_mask)
            center_y = np.mean(y_coords)
            center_x = np.mean(x_coords)
            
            # Compute flows pointing toward center
            for y, x in zip(y_coords, x_coords):
                flows[y, x, 0] = center_y - y  # dY
                flows[y, x, 1] = center_x - x  # dX
        
        return flows, cell_prob
    
    def _apply_augmentations(self, image: np.ndarray, flows: np.ndarray, 
                           cell_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply data augmentations as described in the paper
        """
        # Random rotation and flipping
        if np.random.random() < 0.5:
            # Random rotation
            angle = np.random.uniform(-180, 180)
            image, flows, cell_prob = self._rotate_data(image, flows, cell_prob, angle)
        
        if np.random.random() < 0.5:
            # Random flip
            if np.random.random() < 0.5:  # Horizontal flip
                image = np.fliplr(image)
                flows = np.fliplr(flows)
                flows[..., 1] *= -1  # Flip X component
                cell_prob = np.fliplr(cell_prob)
            else:  # Vertical flip
                image = np.flipud(image)
                flows = np.flipud(flows)
                flows[..., 0] *= -1  # Flip Y component
                cell_prob = np.flipud(cell_prob)
        
        # Random scaling
        if np.random.random() < 0.8:
            scale = np.random.uniform(0.75, 1.25)
            image, flows, cell_prob = self._scale_data(image, flows, cell_prob, scale)
        
        # Color augmentations
        if len(image.shape) == 3:  # Only for color images
            # Random brightness/contrast
            if np.random.random() < 0.5:
                brightness = np.random.normal(0, 0.2)
                contrast = np.random.uniform(0.5, 1.5)
                image = np.clip(image * contrast + brightness, 0, 1)
            
            # Random channel permutation (for channel invariance)
            if np.random.random() < 0.3:
                perm = np.random.permutation(3)
                image = image[..., perm]
            
            # Random grayscale conversion
            if np.random.random() < 0.1:
                gray = np.mean(image, axis=-1, keepdims=True)
                image = np.repeat(gray, 3, axis=-1)
        
        # Contrast inversion
        if np.random.random() < 0.25:
            image = 1.0 - image
        
        return image, flows, cell_prob
    
    def _rotate_data(self, image: np.ndarray, flows: np.ndarray, 
                    cell_prob: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rotate image, flows, and cell probability"""
        H, W = image.shape[:2]
        center = (W // 2, H // 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image
        if len(image.shape) == 3:
            rotated_image = cv2.warpAffine(image, M, (W, H), flags=cv2.INTER_LINEAR)
        else:
            rotated_image = cv2.warpAffine(image, M, (W, H), flags=cv2.INTER_LINEAR)
        
        # Rotate flows (need to rotate the vector components)
        rotated_flows = np.zeros_like(flows)
        cos_a = np.cos(np.radians(angle))
        sin_a = np.sin(np.radians(angle))
        
        # Rotate each flow vector
        rotated_flows[..., 0] = flows[..., 0] * cos_a - flows[..., 1] * sin_a
        rotated_flows[..., 1] = flows[..., 0] * sin_a + flows[..., 1] * cos_a
        
        # Apply spatial rotation to flow field
        rotated_flows = cv2.warpAffine(rotated_flows, M, (W, H), flags=cv2.INTER_LINEAR)
        
        # Rotate cell probability
        rotated_cell_prob = cv2.warpAffine(cell_prob, M, (W, H), flags=cv2.INTER_LINEAR)
        
        return rotated_image, rotated_flows, rotated_cell_prob
    
    def _scale_data(self, image: np.ndarray, flows: np.ndarray, 
                   cell_prob: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Scale image, flows, and cell probability"""
        H, W = image.shape[:2]
        new_H, new_W = int(H * scale), int(W * scale)
        
        # Scale image
        scaled_image = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        
        # Scale flows (vectors scale with the image)
        scaled_flows = cv2.resize(flows, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        scaled_flows *= scale  # Scale the flow magnitudes
        
        # Scale cell probability
        scaled_cell_prob = cv2.resize(cell_prob, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        
        # Crop or pad to original size
        if scale > 1.0:  # Crop
            start_y = (new_H - H) // 2
            start_x = (new_W - W) // 2
            scaled_image = scaled_image[start_y:start_y+H, start_x:start_x+W]
            scaled_flows = scaled_flows[start_y:start_y+H, start_x:start_x+W]
            scaled_cell_prob = scaled_cell_prob[start_y:start_y+H, start_x:start_x+W]
        else:  # Pad
            pad_y = (H - new_H) // 2
            pad_x = (W - new_W) // 2
            if len(scaled_image.shape) == 3:
                scaled_image = np.pad(scaled_image, ((pad_y, H-new_H-pad_y), (pad_x, W-new_W-pad_x), (0, 0)))
            else:
                scaled_image = np.pad(scaled_image, ((pad_y, H-new_H-pad_y), (pad_x, W-new_W-pad_x)))
            scaled_flows = np.pad(scaled_flows, ((pad_y, H-new_H-pad_y), (pad_x, W-new_W-pad_x), (0, 0)))
            scaled_cell_prob = np.pad(scaled_cell_prob, ((pad_y, H-new_H-pad_y), (pad_x, W-new_W-pad_x)))
        
        return scaled_image, scaled_flows, scaled_cell_prob
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] using percentile normalization"""
        # Percentile normalization as used in the paper
        p1 = np.percentile(image, 1)
        p99 = np.percentile(image, 99)
        
        if p99 > p1:
            image = (image - p1) / (p99 - p1)
            image = np.clip(image, 0, 1)
        
        return image


def prepare_dataset_from_folders(image_folder: str, mask_folder: str, 
                                image_ext: str = '.png', mask_ext: str = '.png',
                                mask_suffix: str = '_mask') -> Tuple[List[str], List[str]]:
    """
    Prepare dataset from folders containing images and masks
    
    Args:
        image_folder: Path to folder containing images
        mask_folder: Path to folder containing masks
        image_ext: Image file extension
        mask_ext: Mask file extension
        mask_suffix: Suffix added to image name for corresponding mask
        
    Returns:
        Tuple of (image_paths, mask_paths)
    """
    image_folder = Path(image_folder)
    mask_folder = Path(mask_folder)
    
    image_paths = []
    mask_paths = []
    
    for image_path in sorted(image_folder.glob(f'*{image_ext}')):
        # Find corresponding mask
        image_name = image_path.stem
        mask_name = f"{image_name}{mask_suffix}{mask_ext}"
        mask_path = mask_folder / mask_name
        
        if mask_path.exists():
            image_paths.append(str(image_path))
            mask_paths.append(str(mask_path))
        else:
            print(f"Warning: No mask found for {image_path.name}")
    
    print(f"Found {len(image_paths)} image-mask pairs")
    return image_paths, mask_paths


def create_train_val_split(image_paths: List[str], mask_paths: List[str], 
                          val_fraction: float = 0.2, 
                          random_seed: int = 42) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Split dataset into training and validation sets
    
    Returns:
        train_images, train_masks, val_images, val_masks
    """
    np.random.seed(random_seed)
    
    n_total = len(image_paths)
    n_val = int(n_total * val_fraction)
    
    indices = np.random.permutation(n_total)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_images = [image_paths[i] for i in train_indices]
    train_masks = [mask_paths[i] for i in train_indices]
    val_images = [image_paths[i] for i in val_indices]
    val_masks = [mask_paths[i] for i in val_indices]
    
    print(f"Training set: {len(train_images)} samples")
    print(f"Validation set: {len(val_images)} samples")
    
    return train_images, train_masks, val_images, val_masks


def create_dataloader(image_paths: List[str], mask_paths: List[str],
                     batch_size: int = 8, shuffle: bool = True,
                     num_workers: int = 4, **dataset_kwargs) -> DataLoader:
    """
    Create DataLoader for training
    """
    dataset = CellposeSAMDataset(image_paths, mask_paths, **dataset_kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Example usage
    print("Testing data preparation...")
    
    # Example: prepare dataset from folders
    # image_paths, mask_paths = prepare_dataset_from_folders(
    #     "path/to/images", 
    #     "path/to/masks"
    # )
    
    # # Create train/val split
    # train_imgs, train_masks, val_imgs, val_masks = create_train_val_split(
    #     image_paths, mask_paths, val_fraction=0.2
    # )
    
    # # Create dataloaders
    # train_loader = create_dataloader(train_imgs, train_masks, batch_size=4)
    # val_loader = create_dataloader(val_imgs, val_masks, batch_size=4, shuffle=False)
    
    # print(f"Training batches: {len(train_loader)}")
    # print(f"Validation batches: {len(val_loader)}")
    
    print("Data preparation module ready!")

