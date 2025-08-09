"""
Training Script for Cellpose-SAM

This script implements the training procedure described in the paper:
"Cellpose-SAM: superhuman generalization for cellular segmentation"
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Tuple
import argparse

from cellpose_sam_model import CellposeSAM, cellpose_loss
from data_preparation import (
    prepare_dataset_from_folders, 
    create_train_val_split, 
    create_dataloader
)

class CellposeSAMTrainer:
    """
    Trainer class for Cellpose-SAM model
    """
    
    def __init__(self, 
                 model: CellposeSAM,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cuda',
                 learning_rate: float = 5e-5,
                 weight_decay: float = 0.1,
                 flow_weight: float = 5.0,
                 save_dir: str = 'checkpoints'):
        """
        Initialize trainer
        
        Args:
            model: CellposeSAM model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
            flow_weight: Weight for flow loss relative to probability loss
            save_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.flow_weight = flow_weight
        
        # Create save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Setup optimizer (AdamW as used in paper)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler (as described in paper)
        self.scheduler = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        print(f"Trainer initialized with device: {device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    
    def setup_scheduler(self, total_epochs: int, warmup_epochs: int = 10):
        """
        Setup learning rate scheduler as described in paper:
        - Linear warmup for first 10 epochs
        - Decrease by factor of 10 for last 100 epochs
        - Another factor of 10 for last 50 epochs
        """
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return epoch / warmup_epochs
            elif epoch < total_epochs - 100:
                # Constant learning rate
                return 1.0
            elif epoch < total_epochs - 50:
                # First decay
                return 0.1
            else:
                # Second decay
                return 0.01
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            true_flows = batch['flows'].to(self.device)
            
            # Forward pass
            pred_flows = self.model(images)
            
            # Compute loss
            loss = cellpose_loss(pred_flows, true_flows, self.flow_weight)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                true_flows = batch['flows'].to(self.device)
                
                # Forward pass
                pred_flows = self.model(images)
                
                # Compute loss
                loss = cellpose_loss(pred_flows, true_flows, self.flow_weight)
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def train(self, epochs: int, save_every: int = 50, validate_every: int = 10):
        """
        Main training loop
        
        Args:
            epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            validate_every: Validate every N epochs
        """
        print(f"Starting training for {epochs} epochs...")
        
        # Setup scheduler
        self.setup_scheduler(epochs)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            self.learning_rates.append(current_lr)
            
            # Validate
            if epoch % validate_every == 0:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, 'best_model.pth')
                
                print(f'Epoch {epoch}/{epochs}:')
                print(f'  Train Loss: {train_loss:.4f}')
                print(f'  Val Loss: {val_loss:.4f}')
                print(f'  LR: {current_lr:.2e}')
                print(f'  Time: {time.time() - start_time:.2f}s')
            else:
                print(f'Epoch {epoch}/{epochs}: Train Loss: {train_loss:.4f}, LR: {current_lr:.2e}')
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pth')
        
        # Save final model
        self.save_checkpoint(epochs, 'final_model.pth')
        self.save_training_history()
        
        print("Training completed!")
    
    def save_checkpoint(self, epoch: int, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.save_dir / filename)
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str) -> int:
        """Load model checkpoint and return epoch"""
        checkpoint_path = self.save_dir / filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded: {filename}, epoch {epoch}")
        return epoch
    
    def save_training_history(self):
        """Save training history as JSON and plot"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
        
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            val_epochs = np.arange(0, len(self.train_losses), len(self.train_losses) // len(self.val_losses))[:len(self.val_losses)]
            ax1.plot(val_epochs, self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Learning rate curve
        ax2.plot(self.learning_rates)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Cellpose-SAM model')
    parser.add_argument('--image_folder', type=str, required=True,
                       help='Path to folder containing training images')
    parser.add_argument('--mask_folder', type=str, required=True,
                       help='Path to folder containing training masks')
    parser.add_argument('--sam_checkpoint', type=str, required=True,
                       help='Path to SAM checkpoint (sam_vit_l_0b3195.pth)')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                       help='Weight decay for AdamW optimizer')
    parser.add_argument('--flow_weight', type=float, default=5.0,
                       help='Weight for flow loss relative to probability loss')
    parser.add_argument('--val_fraction', type=float, default=0.2,
                       help='Fraction of data to use for validation')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Prepare dataset
    print("Preparing dataset...")
    image_paths, mask_paths = prepare_dataset_from_folders(
        args.image_folder, 
        args.mask_folder
    )
    
    if len(image_paths) == 0:
        raise ValueError("No image-mask pairs found!")
    
    # Train/val split
    train_images, train_masks, val_images, val_masks = create_train_val_split(
        image_paths, mask_paths, val_fraction=args.val_fraction
    )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_images, train_masks,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        image_size=args.image_size,
        augment=True
    )
    
    val_loader = create_dataloader(
        val_images, val_masks,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        image_size=args.image_size,
        augment=False
    )
    
    # Initialize model
    print("Initializing model...")
    model = CellposeSAM(args.sam_checkpoint, device=args.device)
    
    # Initialize trainer
    trainer = CellposeSAMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        flow_weight=args.flow_weight,
        save_dir=args.save_dir
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(epochs=args.epochs - start_epoch)


if __name__ == "__main__":
    main()

