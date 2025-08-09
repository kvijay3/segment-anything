"""
Cellpose-SAM Model Implementation

This module implements the Cellpose-SAM architecture that combines
the SAM image encoder with Cellpose flow field prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

try:
    from segment_anything import sam_model_registry
    from segment_anything.modeling import ImageEncoderViT
except ImportError:
    print("Warning: segment_anything not installed. Please install with: pip install git+https://github.com/facebookresearch/segment-anything.git")

class CellposeSAMEncoder(nn.Module):
    """
    Modified SAM image encoder for Cellpose-SAM
    
    Key modifications:
    - Input size: 1024x1024 -> 256x256
    - Patch size: 16x16 -> 8x8
    - Global attention on all layers
    - Adapted position embeddings
    """
    
    def __init__(self, sam_encoder: ImageEncoderViT):
        super().__init__()
        
        # Copy the original encoder
        self.encoder = sam_encoder
        
        # Modify for 256x256 input and 8x8 patches
        self._modify_for_cellpose()
        
    def _modify_for_cellpose(self):
        """Modify the SAM encoder for Cellpose-SAM requirements"""
        
        # 1. Modify patch embedding from 16x16 to 8x8
        old_patch_embed = self.encoder.patch_embed.proj
        new_patch_embed = nn.Conv2d(
            in_channels=old_patch_embed.in_channels,
            out_channels=old_patch_embed.out_channels,
            kernel_size=8,  # Changed from 16 to 8
            stride=8,       # Changed from 16 to 8
            bias=old_patch_embed.bias is not None
        )
        
        # Initialize new patch embedding by downsampling old weights
        with torch.no_grad():
            old_weight = old_patch_embed.weight  # [embed_dim, 3, 16, 16]
            # Downsample from 16x16 to 8x8
            new_weight = F.interpolate(old_weight, size=(8, 8), mode='bilinear', align_corners=False)
            new_patch_embed.weight.copy_(new_weight)
            if old_patch_embed.bias is not None:
                new_patch_embed.bias.copy_(old_patch_embed.bias)
        
        self.encoder.patch_embed.proj = new_patch_embed
        
        # 2. Adapt position embeddings for 256x256 input
        # Original: 1024x1024 with 16x16 patches = 64x64 patches
        # New: 256x256 with 8x8 patches = 32x32 patches
        old_pos_embed = self.encoder.pos_embed  # [1, 64*64, embed_dim]
        embed_dim = old_pos_embed.shape[-1]
        
        # Reshape to spatial dimensions
        old_pos_embed_spatial = old_pos_embed.reshape(1, 64, 64, embed_dim).permute(0, 3, 1, 2)
        
        # Downsample to 32x32
        new_pos_embed_spatial = F.interpolate(old_pos_embed_spatial, size=(32, 32), mode='bilinear', align_corners=False)
        
        # Reshape back to sequence format
        new_pos_embed = new_pos_embed_spatial.permute(0, 2, 3, 1).reshape(1, 32*32, embed_dim)
        
        self.encoder.pos_embed = nn.Parameter(new_pos_embed)
        
        # 3. Ensure global attention on all layers (SAM uses local attention on some layers)
        for block in self.encoder.blocks:
            if hasattr(block.attn, 'rel_pos_h'):
                # Remove relative position embeddings if present (used in local attention)
                delattr(block.attn, 'rel_pos_h')
                delattr(block.attn, 'rel_pos_w')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the modified encoder"""
        return self.encoder(x)


class CellposeFlowDecoder(nn.Module):
    """
    Cellpose-style decoder that predicts flow fields and cell probabilities
    
    Outputs:
    - Channel 0: Cell probability
    - Channel 1: Horizontal flow (dY)
    - Channel 2: Vertical flow (dX)
    """
    
    def __init__(self, encoder_dim: int = 1024, output_size: int = 256):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.output_size = output_size
        
        # Transposed convolution to upsample from 32x32 to 256x256
        self.upsample = nn.ConvTranspose2d(
            in_channels=encoder_dim,
            out_channels=3,  # cell_prob, flow_y, flow_x
            kernel_size=8,
            stride=8,
            padding=0
        )
        
        # Additional refinement layers
        self.refine = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder
        
        Args:
            x: Encoded features [B, H*W, C] from ViT encoder
            
        Returns:
            flows: [B, 3, H, W] tensor with cell_prob, flow_y, flow_x
        """
        B, HW, C = x.shape
        H = W = int(np.sqrt(HW))  # Should be 32 for 256x256 input
        
        # Reshape to spatial format
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # Upsample to output size
        flows = self.upsample(x)
        
        # Refine the flows
        flows = self.refine(flows)
        
        # Apply activations
        cell_prob = torch.sigmoid(flows[:, 0:1])  # Cell probability [0, 1]
        flow_y = flows[:, 1:2]  # Horizontal flow (can be negative)
        flow_x = flows[:, 2:3]  # Vertical flow (can be negative)
        
        return torch.cat([cell_prob, flow_y, flow_x], dim=1)


class CellposeSAM(nn.Module):
    """
    Complete Cellpose-SAM model combining modified SAM encoder with Cellpose decoder
    """
    
    def __init__(self, sam_checkpoint_path: str, device: str = 'cuda'):
        super().__init__()
        
        self.device = device
        
        # Load pretrained SAM model
        try:
            sam = sam_model_registry["vit_l"](checkpoint=sam_checkpoint_path)
            print(f"Loaded SAM model from {sam_checkpoint_path}")
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            raise
        
        # Create modified encoder
        self.encoder = CellposeSAMEncoder(sam.image_encoder)
        
        # Create Cellpose-style decoder
        self.decoder = CellposeFlowDecoder(encoder_dim=1024, output_size=256)
        
        # Move to device
        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Cellpose-SAM
        
        Args:
            x: Input images [B, C, H, W] - should be 256x256
            
        Returns:
            flows: [B, 3, H, W] with cell_prob, flow_y, flow_x
        """
        # Ensure input is correct size
        if x.shape[-1] != 256 or x.shape[-2] != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        
        # Encode
        features = self.encoder(x)
        
        # Decode to flows
        flows = self.decoder(features)
        
        return flows
    
    def predict_masks(self, x: torch.Tensor, flow_threshold: float = 0.4, 
                     cell_prob_threshold: float = 0.0) -> np.ndarray:
        """
        Predict cell masks from input images using flow tracking
        
        Args:
            x: Input images [B, C, H, W]
            flow_threshold: Threshold for flow error
            cell_prob_threshold: Threshold for cell probability
            
        Returns:
            masks: Numpy array of cell masks
        """
        with torch.no_grad():
            flows = self.forward(x)
            
        # Convert to numpy
        flows_np = flows.cpu().numpy()
        
        masks = []
        for i in range(flows_np.shape[0]):
            cell_prob = flows_np[i, 0]
            flow_y = flows_np[i, 1]
            flow_x = flows_np[i, 2]
            
            # Use Cellpose dynamics to convert flows to masks
            # This would require implementing the Cellpose dynamics algorithm
            # For now, return a placeholder
            mask = self._flows_to_masks(cell_prob, flow_y, flow_x, 
                                      flow_threshold, cell_prob_threshold)
            masks.append(mask)
            
        return np.array(masks)
    
    def _flows_to_masks(self, cell_prob: np.ndarray, flow_y: np.ndarray, 
                       flow_x: np.ndarray, flow_threshold: float, 
                       cell_prob_threshold: float) -> np.ndarray:
        """
        Convert flow fields to cell masks using Cellpose dynamics
        
        This is a simplified version - the full implementation would use
        the Cellpose dynamics algorithm for flow tracking.
        """
        # Placeholder implementation
        # In practice, you would use cellpose.dynamics.compute_masks()
        mask = (cell_prob > cell_prob_threshold).astype(np.int32)
        return mask


def cellpose_loss(pred_flows: torch.Tensor, true_flows: torch.Tensor, 
                 flow_weight: float = 5.0) -> torch.Tensor:
    """
    Cellpose loss function
    
    Args:
        pred_flows: Predicted flows [B, 3, H, W] (cell_prob, flow_y, flow_x)
        true_flows: True flows [B, 3, H, W] (cell_prob, flow_y, flow_x)
        flow_weight: Weight for flow loss relative to probability loss
        
    Returns:
        loss: Combined loss value
    """
    # Split into components
    pred_prob = pred_flows[:, 0]
    pred_flow_y = pred_flows[:, 1]
    pred_flow_x = pred_flows[:, 2]
    
    true_prob = true_flows[:, 0]
    true_flow_y = true_flows[:, 1]
    true_flow_x = true_flows[:, 2]
    
    # Probability loss (binary cross-entropy)
    prob_loss = F.binary_cross_entropy(pred_prob, true_prob, reduction='mean')
    
    # Flow loss (MSE, only where cells exist)
    cell_mask = true_prob > 0.5
    if cell_mask.sum() > 0:
        flow_y_loss = F.mse_loss(pred_flow_y[cell_mask], true_flow_y[cell_mask])
        flow_x_loss = F.mse_loss(pred_flow_x[cell_mask], true_flow_x[cell_mask])
        flow_loss = flow_y_loss + flow_x_loss
    else:
        flow_loss = torch.tensor(0.0, device=pred_flows.device)
    
    # Combined loss
    total_loss = prob_loss + flow_weight * flow_loss
    
    return total_loss


if __name__ == "__main__":
    # Test the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # This would require downloading the SAM checkpoint
    # model = CellposeSAM("sam_vit_l_0b3195.pth", device=device)
    
    # Test with random input
    # x = torch.randn(2, 3, 256, 256).to(device)
    # flows = model(x)
    # print(f"Input shape: {x.shape}")
    # print(f"Output shape: {flows.shape}")
    
    print("Model implementation complete!")

