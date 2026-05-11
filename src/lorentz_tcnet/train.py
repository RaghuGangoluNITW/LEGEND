"""
Training utilities for Lorentz TCN with LOSO cross-validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List
from pathlib import Path

from .config import TrainConfig
from .data import load_subject_samples, build_loso_splits, TriModalDataset
from .model import TriModalLorentzNet
from .metrics import classification_metrics


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer = None,
    device: str = 'cuda',
    is_training: bool = True
) -> Dict[str, float]:
    """
    Run one epoch of training or validation.
    
    Args:
        model: The model to train/evaluate
        dataloader: DataLoader for the dataset
        criterion: Loss function
        optimizer: Optimizer (only for training)
        device: Device to run on
        is_training: Whether this is a training epoch
        
    Returns:
        Dictionary with 'loss', 'accuracy', 'precision', 'recall', 'f1'
    """
    model.train() if is_training else model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    context = torch.enable_grad() if is_training else torch.no_grad()
    
    with context:
        pbar = tqdm(dataloader, desc='Train' if is_training else 'Val')
        for batch in pbar:
            eeg = batch['eeg'].to(device)
            esg = batch['esg'].to(device)
            emg = batch['emg'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(eeg, esg, emg)
            logits = outputs['logits']
            loss = criterion(logits, labels)
            
            # Backward pass (training only)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    metrics_dict = classification_metrics(
        np.array(all_labels),
        np.array(all_preds)
    )
    metrics_dict['loss'] = avg_loss
    
    return metrics_dict


def train_single_fold(
    train_dataset: TriModalDataset,
    val_dataset: TriModalDataset,
    config: TrainConfig,
    test_subject: str
) -> Dict[str, float]:
    """
    Train and evaluate on a single LOSO fold.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
        test_subject: ID of the test subject
        
    Returns:
        Dictionary with best validation metrics
    """
    print(f"\n{'='*80}")
    print(f"Training fold: Test subject = {test_subject}")
    print(f"{'='*80}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    device = torch.device(config.device)
    model = TriModalLorentzNet(
        eeg_channels=config.eeg_channels,
        esg_channels=config.esg_channels,
        emg_channels=config.emg_channels,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_classes=config.num_classes,
        dropout=config.tcn_dropout
    ).to(device)
    
    # Loss and optimizer
    if config.use_focal_loss:
        criterion = FocalLoss(
            alpha=1.0,
            gamma=config.focal_gamma
        )
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )
    
    # Training loop
    best_val_acc = 0.0
    best_metrics = {}
    patience_counter = 0
    patience = 10  # Early stopping patience
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print("-" * 80)
        
        # Train
        train_metrics = _run_epoch(
            model, train_loader, criterion, optimizer, device, is_training=True
        )
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.2f}%")
        
        # Validate
        val_metrics = _run_epoch(
            model, val_loader, criterion, None, device, is_training=False
        )
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.2f}%, "
              f"F1: {val_metrics['f1']:.2f}%")
        
        # Update scheduler
        scheduler.step(val_metrics['accuracy'])
        
        # Check for improvement
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_metrics = val_metrics.copy()
            patience_counter = 0
            print(f"  ✓ New best accuracy: {best_val_acc:.2f}%")
            
            # Save checkpoint
            checkpoint_dir = Path(config.save_dir) / f"fold_{test_subject}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': best_metrics,
            }, checkpoint_dir / 'best_model.pt')
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    return best_metrics


def run_loso_experiment(config: TrainConfig) -> Dict[str, Dict[str, float]]:
    """
    Run complete LOSO cross-validation experiment.
    
    Args:
        config: Training configuration
        
    Returns:
        Dictionary mapping subject IDs to their validation metrics
    """
    print("="*80)
    print("LORENTZ TCN - Leave-One-Subject-Out Cross-Validation")
    print("="*80)
    
    # Load all subjects
    print(f"\nLoading data from: {config.dataset_dir}")
    subject_samples = load_subject_samples(config.dataset_dir)
    print(f"Loaded {len(subject_samples)} subjects:")
    for subject_id, samples in subject_samples.items():
        print(f"  {subject_id}: {len(samples)} samples")
    
    # Build LOSO splits
    loso_splits = build_loso_splits(subject_samples)
    print(f"\nBuilt {len(loso_splits)} LOSO folds")
    
    # Run each fold
    results = {}
    
    for fold_idx, (train_dataset, val_dataset, test_subject) in enumerate(loso_splits):
        print(f"\n{'#'*80}")
        print(f"# FOLD {fold_idx+1}/{len(loso_splits)}: Test subject = {test_subject}")
        print(f"{'#'*80}")
        
        metrics = train_single_fold(train_dataset, val_dataset, config, test_subject)
        results[test_subject] = metrics
        
        # Print fold summary
        print(f"\nFold {fold_idx+1} Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
        print(f"  Precision: {metrics['precision']:.2f}%")
        print(f"  Recall:    {metrics['recall']:.2f}%")
        print(f"  F1-Score:  {metrics['f1']:.2f}%")
    
    return results
