"""
Training pipeline for Lipschitz-constrained CBF neural network
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import logging

from .lipschitz_network import LipschitzCBFNetwork
from .cbf_loss import CBFLoss


class CBFDataset(Dataset):
    """Dataset for CBF training"""
    
    def __init__(self, states: np.ndarray, labels: np.ndarray, 
                 safe_mask: np.ndarray, boundary_mask: np.ndarray):
        """
        Args:
            states: [N, 38] state samples
            labels: [N] safety labels (1=safe, 0=unsafe)
            safe_mask: [N] boolean mask for safe states
            boundary_mask: [N] boolean mask for boundary states
        """
        self.states = torch.FloatTensor(states)
        self.labels = torch.FloatTensor(labels)
        self.safe_mask = torch.BoolTensor(safe_mask)
        self.boundary_mask = torch.BoolTensor(boundary_mask)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'label': self.labels[idx],
            'safe': self.safe_mask[idx],
            'boundary': self.boundary_mask[idx]
        }


class CBFTrainer:
    """Trainer for Lipschitz CBF network"""
    
    def __init__(self, 
                 model: LipschitzCBFNetwork,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 lr: float = 3e-4,
                 loss_weights: Dict[str, float] = None):
        
        self.model = model.to(device)
        self.device = device
        
        # Default loss weights from paper
        if loss_weights is None:
            loss_weights = {
                'safety': 10.0,
                'validity': 5.0,
                'smoothness': 0.1,
                'cbf_decrease': 2.0
            }
        
        self.criterion = CBFLoss(**loss_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, 
                                    betas=(0.9, 0.999))
        
        # Learning rate scheduler (exponential decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.95
        )
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'safety': 0.0,
            'validity': 0.0,
            'smoothness': 0.0,
            'cbf_decrease': 0.0
        }
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            # Move to device
            states = batch['state'].to(self.device)
            labels = batch['label'].to(self.device)
            safe_mask = batch['safe'].to(self.device)
            
            # Forward pass
            h_pred = self.model(states)
            
            # Compute loss
            losses = self.criterion(h_pred, labels, states, safe_mask, self.model)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            # Update progress bar
            pbar.set_postfix({'loss': losses['total'].item()})
        
        # Average losses
        n_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        epoch_losses = {
            'total': 0.0,
            'safety': 0.0,
            'validity': 0.0,
            'smoothness': 0.0,
            'cbf_decrease': 0.0
        }
        
        # Classification metrics
        correct = 0
        total = 0
        
        for batch in val_loader:
            states = batch['state'].to(self.device)
            labels = batch['label'].to(self.device)
            safe_mask = batch['safe'].to(self.device)
            
            h_pred = self.model(states)
            
            # Compute losses
            losses = self.criterion(h_pred, labels, states, safe_mask, self.model)
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            # Classification accuracy
            predictions = (h_pred.squeeze() > 0).float()
            correct += (predictions == labels).sum().item()
            total += len(labels)
        
        # Average
        n_batches = len(val_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        epoch_losses['accuracy'] = correct / total
        
        return epoch_losses
    
    def fit(self, 
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 500,
            save_path: Optional[str] = None,
            early_stopping_patience: int = 50):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            save_path: Path to save best model
            early_stopping_patience: Epochs without improvement before stopping
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_losses = self.train_epoch(train_loader)
            self.train_history.append(train_losses)
            
            # Validate
            val_losses = self.validate(val_loader)
            self.val_history.append(val_losses)
            
            # Learning rate scheduling
            if (epoch + 1) % 100 == 0:
                self.scheduler.step()
            
            # Logging
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_losses['total']:.4f} | "
                f"Val Loss: {val_losses['total']:.4f} | "
                f"Val Acc: {val_losses['accuracy']:.4f} | "
                f"Lipschitz: {self.model.lipschitz_constant():.4f}"
            )
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                
                if save_path:
                    self.save_checkpoint(save_path, epoch, val_losses)
                    self.logger.info(f"Saved best model to {save_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.train_history, self.val_history
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'lipschitz_constant': self.model.lipschitz_constant()
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        self.logger.info(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} "
            f"with Lipschitz constant {checkpoint['lipschitz_constant']:.4f}"
        )