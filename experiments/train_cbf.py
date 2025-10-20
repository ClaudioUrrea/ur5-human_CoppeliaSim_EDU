"""
Training script for CBF neural network
"""

import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.cbf.lipschitz_network import LipschitzCBFNetwork
from src.cbf.cbf_trainer import CBFTrainer, CBFDataset


def setup_logging(log_dir: Path):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )


def load_data(data_path: Path):
    """Load training data"""
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    
    states = data['states']
    labels = data['labels']
    safe_mask = data['safe_mask']
    boundary_mask = data['boundary_mask']
    
    print(f"Loaded {len(states)} samples")
    print(f"  State dimension: {states.shape[1]}")
    print(f"  Safe samples: {np.sum(safe_mask)}")
    print(f"  Unsafe samples: {np.sum(~safe_mask)}")
    
    return states, labels, safe_mask, boundary_mask


def main():
    parser = argparse.ArgumentParser(description='Train CBF neural network')
    parser.add_argument('--data', type=str, 
                       default='data/raw/training_dataset_50k.npz',
                       help='Path to training data')
    parser.add_argument('--output', type=str,
                       default='models/checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load data
    states, labels, safe_mask, boundary_mask = load_data(Path(args.data))
    
    # Create dataset
    dataset = CBFDataset(states, labels, safe_mask, boundary_mask)
    
    # Train/val split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == 'cuda')
    )
    
    # Create model
    print("Creating model...")
    model = LipschitzCBFNetwork(
        input_dim=38,
        hidden_dims=[128, 64, 32],
        L_max=1.0
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Lipschitz constant: {model.lipschitz_constant():.4f}")
    
    # Create trainer
    loss_weights = {
        'safety': 10.0,
        'validity': 5.0,
        'smoothness': 0.1,
        'cbf_decrease': 2.0
    }
    
    trainer = CBFTrainer(
        model=model,
        device=device,
        lr=args.lr,
        loss_weights=loss_weights
    )
    
    # Save training config
    config = {
        'model': {
            'input_dim': 38,
            'hidden_dims': [128, 64, 32],
            'L_max': 1.0
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'loss_weights': loss_weights
        },
        'data': {
            'total_samples': len(dataset),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset)
        }
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)
    
    train_history, val_history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_path=output_dir / 'cbf_lipschitz_best.pth',
        early_stopping_patience=50
    )
    
    # Save final model
    trainer.save_checkpoint(
        output_dir / 'cbf_lipschitz_final.pth',
        epoch=args.epochs,
        metrics=val_history[-1] if val_history else {}
    )
    
    # Save training history
    np.savez(
        output_dir / 'training_history.npz',
        train_loss=[h['total'] for h in train_history],
        val_loss=[h['total'] for h in val_history],
        val_accuracy=[h['accuracy'] for h in val_history]
    )
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best model saved to: {output_dir / 'cbf_lipschitz_best.pth'}")
    print(f"Final validation accuracy: {val_history[-1]['accuracy']:.4f}")
    print(f"Final Lipschitz constant: {model.lipschitz_constant():.4f}")


if __name__ == "__main__":
    main()