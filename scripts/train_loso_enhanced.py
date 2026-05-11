#!/usr/bin/env python3
"""
Phase 3 Enhanced Training - Steele Dataset
===========================================

Trains EnhancedTriModalLorentzNet with:
- Spatial-temporal attention
- Data augmentation
- Improved fusion layers

PRESERVES Lorentzian core while adding complementary enhancements.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path

from src.lorentz_tcnet.model_enhanced import EnhancedTriModalLorentzNet, TimeSeriesAugmentation


class TriModalDataset(Dataset):
    def __init__(self, eeg, esg, emg, labels, augment=False):
        self.eeg = torch.FloatTensor(eeg)
        self.esg = torch.FloatTensor(esg)
        self.emg = torch.FloatTensor(emg)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        eeg = self.eeg[idx]
        esg = self.esg[idx]
        emg = self.emg[idx]
        
        # Apply augmentation during training
        if self.augment:
            eeg = TimeSeriesAugmentation.apply_augmentations(eeg.unsqueeze(0), aug_prob=0.5).squeeze(0)
            esg = TimeSeriesAugmentation.apply_augmentations(esg.unsqueeze(0), aug_prob=0.5).squeeze(0)
            emg = TimeSeriesAugmentation.apply_augmentations(emg.unsqueeze(0), aug_prob=0.5).squeeze(0)
        
        return eeg, esg, emg, self.labels[idx]


def load_subject_data(data_dir, subject_id):
    """Load tri-modal data for a single subject."""
    data_path = data_dir / f"{subject_id}.npz"
    
    data = np.load(data_path)
    eeg = data['eeg']
    esg = data['esg']
    emg = data['emg']
    labels = data['labels']
    
    return eeg, esg, emg, labels


def train_one_subject(subject_id, all_subjects, data_dir, device, num_epochs=50):
    """Train with LOSO - one subject as test, rest as train."""
    
    print(f"\n{'='*80}")
    print(f"Test Subject: {subject_id}")
    print('='*80)
    
    # Split train/test
    train_subjects = [s for s in all_subjects if s != subject_id]
    
    # Load training data
    train_eeg, train_esg, train_emg, train_labels = [], [], [], []
    for subj in train_subjects:
        eeg, esg, emg, labels = load_subject_data(data_dir, subj)
        train_eeg.append(eeg)
        train_esg.append(esg)
        train_emg.append(emg)
        train_labels.append(labels)
    
    train_eeg = np.concatenate(train_eeg, axis=0)
    train_esg = np.concatenate(train_esg, axis=0)
    train_emg = np.concatenate(train_emg, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    
    # Load test data
    test_eeg, test_esg, test_emg, test_labels = load_subject_data(data_dir, subject_id)
    
    print(f"Train: {len(train_labels)} trials")
    print(f"Test:  {len(test_labels)} trials")
    
    # Create datasets with augmentation for training
    train_dataset = TriModalDataset(train_eeg, train_esg, train_emg, train_labels, augment=True)
    test_dataset = TriModalDataset(test_eeg, test_esg, test_emg, test_labels, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Phase 3 Enhanced Model
    n_eeg_channels = train_eeg.shape[1]
    n_esg_channels = train_esg.shape[1]
    n_emg_channels = train_emg.shape[1]
    
    model = EnhancedTriModalLorentzNet(
        eeg_channels=n_eeg_channels,
        esg_channels=n_esg_channels,
        emg_channels=n_emg_channels,
        hidden_dim=64,
        latent_dim=32,
        num_classes=4,
        dropout=0.5,
        use_attention=True  # Enable attention
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    best_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_correct = 0
        train_total = 0
        
        for batch_eeg, batch_esg, batch_emg, batch_y in train_loader:
            batch_eeg = batch_eeg.to(device)
            batch_esg = batch_esg.to(device)
            batch_emg = batch_emg.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_eeg, batch_esg, batch_emg)
            logits = outputs['logits']
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            _, predicted = logits.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        
        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_eeg, batch_esg, batch_emg, batch_y in test_loader:
                batch_eeg = batch_eeg.to(device)
                batch_esg = batch_esg.to(device)
                batch_emg = batch_emg.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_eeg, batch_esg, batch_emg)
                logits = outputs['logits']
                _, predicted = logits.max(1)
                test_total += batch_y.size(0)
                test_correct += predicted.eq(batch_y).sum().item()
        
        test_acc = 100.0 * test_correct / test_total
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        scheduler.step(test_acc)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d}: Train={train_acc:.2f}% | Test={test_acc:.2f}% (Best={best_acc:.2f}%)")
    
    print(f"\n✅ {subject_id} Best: {best_acc:.2f}%")
    
    return {
        'subject': subject_id,
        'accuracy': best_acc,
        'n_train': len(train_labels),
        'n_test': len(test_labels)
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data directory
    data_dir = Path(__file__).parent.parent / 'data' / 'steele'
    
    # All subjects
    all_subjects = [
        'NIS001', 'NIS002', 'NIS003', 'NIS004', 'NIS005',
        'NIS006', 'NIS007', 'NIS008', 'NIS009', 'NIS010'
    ]
    
    print("="*80)
    print("Phase 3 Enhanced Training - Steele Dataset (LOSO)")
    print("="*80)
    print(f"Subjects: {', '.join(all_subjects)}")
    print(f"Total: {len(all_subjects)} subjects")
    print("="*80)
    
    results = []
    
    for subject_id in tqdm(all_subjects, desc="LOSO Subjects"):
        result = train_one_subject(
            subject_id=subject_id,
            all_subjects=all_subjects,
            data_dir=data_dir,
            device=device,
            num_epochs=50
        )
        results.append(result)
    
    # Calculate statistics
    accuracies = [r['accuracy'] for r in results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)
    
    print("\n" + "="*80)
    print("FINAL RESULTS - Steele Dataset (Phase 3 Enhanced)")
    print("="*80)
    print(f"\nMean Accuracy: {mean_acc:.2f}%")
    print(f"Std Accuracy:  {std_acc:.2f}%")
    print(f"Min Accuracy:  {min_acc:.2f}%")
    print(f"Max Accuracy:  {max_acc:.2f}%")
    
    # Save results
    output_path = Path(__file__).parent.parent / 'steele_results_enhanced.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Results saved to: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
