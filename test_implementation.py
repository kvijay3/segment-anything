"""
Test script for Cellpose-SAM implementation

This script tests the basic functionality without requiring
the full SAM checkpoint or training data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def test_model_architecture():
    """Test the model architecture without SAM weights"""
    print("Testing model architecture...")
    
    try:
        from cellpose_sam_model import CellposeFlowDecoder, cellpose_loss
        
        # Test decoder
        decoder = CellposeFlowDecoder(encoder_dim=1024, output_size=256)
        
        # Create dummy input (simulating ViT output)
        batch_size = 2
        seq_len = 32 * 32  # 32x32 patches for 256x256 image
        embed_dim = 1024
        
        dummy_features = torch.randn(batch_size, seq_len, embed_dim)
        
        # Forward pass
        flows = decoder(dummy_features)
        
        print(f"Input shape: {dummy_features.shape}")
        print(f"Output shape: {flows.shape}")
        print(f"Expected output shape: [2, 3, 256, 256]")
        
        assert flows.shape == (batch_size, 3, 256, 256), f"Wrong output shape: {flows.shape}"
        
        # Test loss function
        true_flows = torch.randn_like(flows)
        true_flows[:, 0] = torch.sigmoid(true_flows[:, 0])  # Cell prob should be [0,1]
        
        loss = cellpose_loss(flows, true_flows)
        print(f"Loss value: {loss.item():.4f}")
        
        print("✓ Model architecture test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Model architecture test failed: {e}")
        return False

def test_data_preparation():
    """Test data preparation functionality"""
    print("\nTesting data preparation...")
    
    try:
        from data_preparation import CellposeSAMDataset
        
        # Create dummy data
        dummy_dir = Path("test_data")
        dummy_dir.mkdir(exist_ok=True)
        
        images_dir = dummy_dir / "images"
        masks_dir = dummy_dir / "masks"
        images_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        
        # Create dummy image and mask
        dummy_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        dummy_mask = np.zeros((128, 128), dtype=np.int32)
        
        # Add some dummy cells to mask
        dummy_mask[30:50, 30:50] = 1
        dummy_mask[70:90, 70:90] = 2
        
        # Save dummy data
        import cv2
        cv2.imwrite(str(images_dir / "test_image.png"), dummy_image)
        cv2.imwrite(str(masks_dir / "test_image_mask.png"), dummy_mask)
        
        # Test dataset
        dataset = CellposeSAMDataset(
            image_paths=[str(images_dir / "test_image.png")],
            mask_paths=[str(masks_dir / "test_image_mask.png")],
            image_size=256,
            augment=False
        )
        
        sample = dataset[0]
        
        print(f"Image shape: {sample['image'].shape}")
        print(f"Flows shape: {sample['flows'].shape}")
        
        assert sample['image'].shape[1:] == (256, 256), f"Wrong image shape: {sample['image'].shape}"
        assert sample['flows'].shape == (3, 256, 256), f"Wrong flows shape: {sample['flows'].shape}"
        
        # Cleanup
        import shutil
        shutil.rmtree(dummy_dir)
        
        print("✓ Data preparation test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Data preparation test failed: {e}")
        return False

def test_training_components():
    """Test training components without actual training"""
    print("\nTesting training components...")
    
    try:
        from train_cellpose_sam import CellposeSAMTrainer
        from cellpose_sam_model import CellposeFlowDecoder
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy model (just the decoder for testing)
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = CellposeFlowDecoder()
                
            def forward(self, x):
                # Simulate encoder output
                B, C, H, W = x.shape
                features = torch.randn(B, (H//8) * (W//8), 1024)
                return self.decoder(features)
        
        model = DummyModel()
        
        # Create dummy data
        dummy_images = torch.randn(10, 3, 256, 256)
        dummy_flows = torch.randn(10, 3, 256, 256)
        dummy_flows[:, 0] = torch.sigmoid(dummy_flows[:, 0])  # Cell prob
        
        dataset = TensorDataset(dummy_images, dummy_flows)
        train_loader = DataLoader(dataset, batch_size=2)
        val_loader = DataLoader(dataset, batch_size=2)
        
        # Test trainer initialization
        trainer = CellposeSAMTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu'  # Use CPU for testing
        )
        
        # Test one training step
        model.train()
        batch = next(iter(train_loader))
        images, true_flows = batch
        
        pred_flows = model(images)
        loss = trainer.flow_weight * torch.nn.functional.mse_loss(pred_flows, true_flows)
        
        print(f"Training step test - Loss: {loss.item():.4f}")
        
        print("✓ Training components test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Training components test failed: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    print("\nCreating sample data for testing...")
    
    try:
        # Create directories
        data_dir = Path("sample_data")
        data_dir.mkdir(exist_ok=True)
        
        images_dir = data_dir / "images"
        masks_dir = data_dir / "masks"
        images_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        
        # Create synthetic cell images
        for i in range(5):
            # Create image with synthetic cells
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.int32)
            
            # Add random circular cells
            num_cells = np.random.randint(3, 8)
            cell_id = 1
            
            for _ in range(num_cells):
                # Random center and radius
                center_x = np.random.randint(30, 226)
                center_y = np.random.randint(30, 226)
                radius = np.random.randint(10, 25)
                
                # Create circular cell
                y, x = np.ogrid[:256, :256]
                cell_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                
                # Avoid overlap
                if not np.any(mask[cell_mask] > 0):
                    # Add to image (bright cell with darker border)
                    image[cell_mask] = [200, 200, 200]  # Cell interior
                    
                    # Add border
                    border_mask = ((x - center_x)**2 + (y - center_y)**2 <= (radius+2)**2) & \
                                 ((x - center_x)**2 + (y - center_y)**2 > radius**2)
                    image[border_mask] = [100, 100, 100]  # Cell border
                    
                    # Add to mask
                    mask[cell_mask] = cell_id
                    cell_id += 1
            
            # Add some noise
            noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
            image = np.clip(image.astype(np.int32) + noise, 0, 255).astype(np.uint8)
            
            # Save
            import cv2
            cv2.imwrite(str(images_dir / f"sample_{i:03d}.png"), image)
            cv2.imwrite(str(masks_dir / f"sample_{i:03d}_mask.png"), mask)
        
        print(f"✓ Created 5 sample images in {data_dir}/")
        print("You can use these for testing with:")
        print(f"python train_cellpose_sam.py --image_folder {images_dir} --mask_folder {masks_dir} --sam_checkpoint sam_vit_l_0b3195.pth --epochs 10")
        
        return True
        
    except Exception as e:
        print(f"✗ Sample data creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("CELLPOSE-SAM IMPLEMENTATION TEST")
    print("=" * 60)
    
    tests = [
        test_model_architecture,
        test_data_preparation,
        test_training_components,
        create_sample_data
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"{test.__name__}: {status}")
    
    if all(results):
        print("\n✓ All tests passed! Implementation is ready.")
        print("\nNext steps:")
        print("1. Download SAM weights: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth")
        print("2. Prepare your training data or use the sample data created")
        print("3. Run training script")
    else:
        print("\n✗ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()

