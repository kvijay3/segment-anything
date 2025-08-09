# Dataset Preparation Guide for Cellpose-SAM

This guide explains how to prepare your image and mask data for training the Cellpose-SAM model.

## Data Format Requirements

### Images
- **Format**: PNG, TIFF, or JPG
- **Channels**: 1-3 channels
  - Grayscale: Single channel microscopy images
  - RGB: Color images or multi-channel fluorescence
  - Multi-channel: Up to 3 channels (e.g., DAPI + cytoplasm + membrane)
- **Size**: Any size (will be resized to 256x256 during training)
- **Bit depth**: 8-bit or 16-bit

### Masks
- **Format**: PNG or TIFF (preferably PNG for smaller file size)
- **Values**: Integer labels where:
  - `0` = background
  - `1, 2, 3, ...` = individual cell IDs
- **Important**: Each cell must have a unique integer ID
- **Size**: Must match corresponding image dimensions

## Directory Structure

Organize your data as follows:

```
your_dataset/
├── images/
│   ├── img001.png
│   ├── img002.png
│   ├── img003.png
│   └── ...
├── masks/
│   ├── img001_mask.png
│   ├── img002_mask.png
│   ├── img003_mask.png
│   └── ...
```

**Naming Convention**: 
- Images: `{name}.{ext}`
- Masks: `{name}_mask.{ext}`

## Converting Your Data

### From Binary Masks

If you have binary masks (0=background, 255=cell), convert them to instance masks:

```python
import cv2
import numpy as np
from scipy import ndimage

def binary_to_instance_mask(binary_mask):
    """Convert binary mask to instance mask with unique cell IDs"""
    
    # Ensure binary
    binary_mask = (binary_mask > 0).astype(np.uint8)
    
    # Label connected components
    labeled_mask, num_labels = ndimage.label(binary_mask)
    
    return labeled_mask.astype(np.int32)

# Example usage
binary_mask = cv2.imread('binary_mask.png', cv2.IMREAD_GRAYSCALE)
instance_mask = binary_to_instance_mask(binary_mask)
cv2.imwrite('instance_mask.png', instance_mask)
```

### From Contours/Polygons

If you have contour annotations (e.g., from ImageJ, CVAT, or LabelMe):

```python
import cv2
import numpy as np

def contours_to_mask(image_shape, contours):
    """Convert list of contours to instance mask"""
    
    mask = np.zeros(image_shape[:2], dtype=np.int32)
    
    for i, contour in enumerate(contours):
        # Fill contour with unique ID
        cv2.fillPoly(mask, [contour], i + 1)
    
    return mask

# Example usage
# contours = [...] # List of contours as numpy arrays
# mask = contours_to_mask((height, width), contours)
```

### From COCO Format

If you have COCO-style annotations:

```python
import json
import numpy as np
from pycocotools import mask as coco_mask

def coco_to_instance_mask(coco_annotation, image_height, image_width):
    """Convert COCO annotation to instance mask"""
    
    instance_mask = np.zeros((image_height, image_width), dtype=np.int32)
    
    for i, ann in enumerate(coco_annotation['annotations']):
        if 'segmentation' in ann:
            # Convert RLE or polygon to mask
            if isinstance(ann['segmentation'], list):
                # Polygon format
                rles = coco_mask.frPyObjects(ann['segmentation'], image_height, image_width)
                rle = coco_mask.merge(rles)
            else:
                # RLE format
                rle = ann['segmentation']
            
            binary_mask = coco_mask.decode(rle)
            instance_mask[binary_mask > 0] = i + 1
    
    return instance_mask
```

## Data Quality Checks

### Check Mask Quality

```python
import numpy as np
import matplotlib.pyplot as plt

def check_mask_quality(mask_path, image_path=None):
    """Check mask quality and visualize"""
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    print(f"Mask shape: {mask.shape}")
    print(f"Unique values: {np.unique(mask)}")
    print(f"Number of cells: {len(np.unique(mask)) - 1}")  # Exclude background
    print(f"Background pixels: {np.sum(mask == 0)}")
    print(f"Cell pixels: {np.sum(mask > 0)}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2 if image_path else 1, figsize=(12, 6))
    
    if image_path:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='tab20')
        axes[1].set_title(f'Mask ({len(np.unique(mask))-1} cells)')
        axes[1].axis('off')
    else:
        axes.imshow(mask, cmap='tab20')
        axes.set_title(f'Mask ({len(np.unique(mask))-1} cells)')
        axes.axis('off')
    
    plt.tight_layout()
    plt.show()

# Usage
check_mask_quality('path/to/mask.png', 'path/to/image.png')
```

### Validate Dataset

```python
def validate_dataset(image_folder, mask_folder):
    """Validate entire dataset"""
    
    from pathlib import Path
    
    image_folder = Path(image_folder)
    mask_folder = Path(mask_folder)
    
    image_files = list(image_folder.glob('*.png')) + list(image_folder.glob('*.tiff'))
    
    issues = []
    valid_pairs = 0
    
    for img_path in image_files:
        # Find corresponding mask
        mask_name = f"{img_path.stem}_mask.png"
        mask_path = mask_folder / mask_name
        
        if not mask_path.exists():
            issues.append(f"Missing mask for {img_path.name}")
            continue
        
        # Load and check
        try:
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                issues.append(f"Cannot load image: {img_path.name}")
                continue
            
            if mask is None:
                issues.append(f"Cannot load mask: {mask_path.name}")
                continue
            
            # Check dimensions
            if img.shape[:2] != mask.shape:
                issues.append(f"Size mismatch: {img_path.name} {img.shape[:2]} vs {mask.shape}")
                continue
            
            # Check mask values
            unique_vals = np.unique(mask)
            if len(unique_vals) < 2:  # Should have background + at least 1 cell
                issues.append(f"No cells in mask: {mask_path.name}")
                continue
            
            valid_pairs += 1
            
        except Exception as e:
            issues.append(f"Error processing {img_path.name}: {e}")
    
    print(f"Dataset validation complete:")
    print(f"Valid pairs: {valid_pairs}")
    print(f"Issues found: {len(issues)}")
    
    if issues:
        print("\nIssues:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    
    return valid_pairs, issues

# Usage
valid_pairs, issues = validate_dataset('data/images', 'data/masks')
```

## Common Data Issues and Solutions

### Issue 1: Overlapping Cells
**Problem**: Multiple cells have the same ID or overlapping regions.
**Solution**: Use watershed segmentation or manual correction.

```python
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_maxima

def separate_overlapping_cells(binary_mask):
    """Separate overlapping cells using watershed"""
    
    # Distance transform
    distance = ndimage.distance_transform_edt(binary_mask)
    
    # Find peaks (cell centers)
    peaks = peak_local_maxima(distance, min_distance=10, threshold_abs=5)
    markers = np.zeros_like(binary_mask, dtype=int)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    
    # Watershed segmentation
    labels = watershed(-distance, markers, mask=binary_mask)
    
    return labels
```

### Issue 2: Cells Touching Image Borders
**Problem**: Cells cut off at image edges.
**Solution**: Either exclude them or pad the image.

```python
def remove_border_cells(mask):
    """Remove cells touching image borders"""
    
    # Find cells touching borders
    border_cells = set()
    H, W = mask.shape
    
    # Check all borders
    border_cells.update(np.unique(mask[0, :]))    # Top
    border_cells.update(np.unique(mask[-1, :]))   # Bottom
    border_cells.update(np.unique(mask[:, 0]))    # Left
    border_cells.update(np.unique(mask[:, -1]))   # Right
    
    border_cells.discard(0)  # Remove background
    
    # Remove border cells
    clean_mask = mask.copy()
    for cell_id in border_cells:
        clean_mask[mask == cell_id] = 0
    
    # Relabel remaining cells
    unique_ids = np.unique(clean_mask)
    unique_ids = unique_ids[unique_ids > 0]
    
    final_mask = np.zeros_like(clean_mask)
    for new_id, old_id in enumerate(unique_ids, 1):
        final_mask[clean_mask == old_id] = new_id
    
    return final_mask
```

### Issue 3: Very Small or Large Cells
**Problem**: Cells that are too small (noise) or too large (merged cells).
**Solution**: Filter by size.

```python
def filter_cells_by_size(mask, min_size=50, max_size=10000):
    """Filter cells by size"""
    
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids > 0]
    
    filtered_mask = np.zeros_like(mask)
    new_id = 1
    
    for cell_id in unique_ids:
        cell_pixels = np.sum(mask == cell_id)
        
        if min_size <= cell_pixels <= max_size:
            filtered_mask[mask == cell_id] = new_id
            new_id += 1
    
    return filtered_mask
```

## Dataset Statistics

After preparing your dataset, compute statistics:

```python
def compute_dataset_stats(image_folder, mask_folder):
    """Compute dataset statistics"""
    
    from pathlib import Path
    
    stats = {
        'num_images': 0,
        'total_cells': 0,
        'cells_per_image': [],
        'cell_sizes': [],
        'image_sizes': []
    }
    
    image_folder = Path(image_folder)
    mask_folder = Path(mask_folder)
    
    for img_path in image_folder.glob('*.png'):
        mask_path = mask_folder / f"{img_path.stem}_mask.png"
        
        if not mask_path.exists():
            continue
        
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        # Count cells
        unique_ids = np.unique(mask)
        num_cells = len(unique_ids) - 1  # Exclude background
        
        stats['num_images'] += 1
        stats['total_cells'] += num_cells
        stats['cells_per_image'].append(num_cells)
        stats['image_sizes'].append(mask.shape)
        
        # Cell sizes
        for cell_id in unique_ids[1:]:  # Skip background
            cell_size = np.sum(mask == cell_id)
            stats['cell_sizes'].append(cell_size)
    
    # Compute summary statistics
    stats['avg_cells_per_image'] = np.mean(stats['cells_per_image'])
    stats['avg_cell_size'] = np.mean(stats['cell_sizes'])
    stats['median_cell_size'] = np.median(stats['cell_sizes'])
    
    print("Dataset Statistics:")
    print(f"  Images: {stats['num_images']}")
    print(f"  Total cells: {stats['total_cells']}")
    print(f"  Avg cells per image: {stats['avg_cells_per_image']:.1f}")
    print(f"  Avg cell size: {stats['avg_cell_size']:.1f} pixels")
    print(f"  Median cell size: {stats['median_cell_size']:.1f} pixels")
    
    return stats

# Usage
stats = compute_dataset_stats('data/images', 'data/masks')
```

## Training Data Recommendations

Based on the paper, for best results:

1. **Minimum dataset size**: 100+ image-mask pairs
2. **Recommended size**: 500+ pairs for good generalization
3. **Cell diversity**: Include various cell types, sizes, and imaging conditions
4. **Image quality**: Mix of high and low quality images
5. **Annotation quality**: Ensure accurate cell boundaries

## Next Steps

After preparing your dataset:

1. Run validation: `python -c "from DATASET_PREPARATION_GUIDE import validate_dataset; validate_dataset('data/images', 'data/masks')"`
2. Compute statistics to understand your data
3. Start training with the prepared dataset
4. Monitor training progress and adjust hyperparameters as needed

For questions about specific data formats or conversion issues, refer to the main README or create an issue in the repository.

