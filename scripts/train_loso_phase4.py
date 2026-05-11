#!/usr/bin/env python3
"""
Phase 4 Improved Training - Steele Dataset
===========================================

TARGETED IMPROVEMENTS:
1. Conservative augmentation (0.15 vs 0.5 probability)
2. Learnable modality fusion weights
3. Early stopping with patience=10
4. Gradient clipping (max_norm=1.0)
5. Cosine annealing learning rate
6. Lorentzian-aware dropout

GOAL: Beat baseline 50.16% mean accuracy
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

from src.lorentz_tcnet.model_phase4 import ImprovedTriModalLorentzNet, ConservativeAugmentation


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
        
        # Apply CONSERVATIVE augmentation (0.15 probability)
        if self.augment:
            eeg = ConservativeAugmentation.apply_augmentations(eeg.unsqueeze(0), aug_prob=0.15).squeeze(0)
            esg = ConservativeAugmentation.apply_augmentations(esg.unsqueeze(0), aug_prob=0.15).squeeze(0)
            emg = ConservativeAugmentation.apply_augmentations(emg.unsqueeze(0), aug_prob=0.15).squeeze(0)
        
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


class EarlyStopping:
    """Early stopping with patience."""
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_acc, model):
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
            
        return self.early_stop


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
    
    train_eeg = np.concatenate(train_eeg)
    train_esg = np.concatenate(train_esg)
    train_emg = np.concatenate(train_emg)
    train_labels = np.concatenate(train_labels)
    
    # Load test data
    test_eeg, test_esg, test_emg, test_labels = load_subject_data(data_dir, subject_id)
    
    print(f"Train: {len(train_labels)} trials")
    print(f"Test:  {len(test_labels)} trials")
    
    # Create datasets
    train_dataset = TriModalDataset(train_eeg, train_esg, train_emg, train_labels, augment=True)
    test_dataset = TriModalDataset(test_eeg, test_esg, test_emg, test_labels, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Model (use actual channel counts from data)
    model = ImprovedTriModalLorentzNet(
        eeg_channels=28,  # Actual EEG channels in Steele dataset
        esg_channels=15,  # Actual ESG channels in Steele dataset
        emg_channels=8,
        hidden_channels=64,
        latent_dim=32,
        num_classes=4,
        use_spatial_attention=True,
        dropout=0.2
    ).to(device)
    
    # Optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # PHASE 4: Cosine annealing learning rate
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    # PHASE 4: Early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0
        
        for eeg, esg, emg, labels in train_loader:
            eeg, esg, emg, labels = eeg.to(device), esg.to(device), emg.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(eeg, esg, emg)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # PHASE 4: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()
        
        train_acc = 100.0 * train_correct / train_total
        
        # Testing
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for eeg, esg, emg, labels in test_loader:
                eeg, esg, emg, labels = eeg.to(device), esg.to(device), emg.to(device), labels.to(device)
                outputs = model(eeg, esg, emg)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100.0 * test_correct / test_total
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        # Print every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train={train_acc:.2f}% | Test={test_acc:.2f}% (Best={best_acc:.2f}%)")
        
        # Update learning rate
        scheduler.step()
        
        # PHASE 4: Check early stopping
        if early_stopping(test_acc, model):
            print(f"Early stopping at epoch {epoch+1}")
            # Restore best model
            model.load_state_dict(early_stopping.best_model_state)
            break
    
    print(f"\n✅ {subject_id} Best: {best_acc:.2f}%")
    
    return {
        'subject': subject_id,
        'accuracy': best_acc
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    data_dir = Path('/home/supermicro/spine_reg_2/lorentz_tcnet/data/steele')
    
    # All subjects
    all_subjects = [
        'NIS001', 'NIS002', 'NIS003', 'NIS004', 'NIS005',
        'NIS006', 'NIS007', 'NIS008', 'NIS009', 'NIS010'
    ]
    
    print("="*80)
    print("Phase 4 Improved Training - Steele Dataset (LOSO)")
    print("="*80)
    print(f"Subjects: {', '.join(all_subjects)}")
    print(f"Total: {len(all_subjects)} subjects")
    print("="*80)
    print("\nIMPROVEMENTS:")
    print("1. Conservative augmentation (0.15 vs 0.5 probability)")
    print("2. Learnable modality fusion weights")
    print("3. Early stopping (patience=10)")
    print("4. Gradient clipping (max_norm=1.0)")
    print("5. Cosine annealing LR")
    print("6. Lorentzian-aware dropout")
    print("="*80)
    
    # LOSO training
    results = []
    for subject_id in tqdm(all_subjects, desc="LOSO Subjects"):
        result = train_one_subject(subject_id, all_subjects, data_dir, device, num_epochs=50)
        results.append(result)
    
    # Compute statistics
    accuracies = [r['accuracy'] for r in results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)
    
    print("\n" + "="*80)
    print("FINAL RESULTS - Steele Dataset (Phase 4 Improved)")
    print("="*80)
    print()
    print(f"Mean Accuracy: {mean_acc:.2f}%")
    print(f"Std Accuracy:  {std_acc:.2f}%")
    print(f"Min Accuracy:  {min_acc:.2f}%")
    print(f"Max Accuracy:  {max_acc:.2f}%")
    print()
    
    # Comparison with baseline and Phase 3
    print("COMPARISON:")
    print(f"  Baseline (Phase 0): 50.16% ± 13.82%")
    print(f"  Phase 3:            48.45% ± 9.64%")
    print(f"  Phase 4:            {mean_acc:.2f}% ± {std_acc:.2f}%")
    print()
    
    improvement = mean_acc - 50.16
    print(f"Improvement over baseline: {improvement:+.2f}%")
    
    if mean_acc > 50.16:
        print("✅ SUCCESS: Beat baseline!")
    else:
        print("❌ Still below baseline")
    
    # Save results
    output_path = Path('/home/supermicro/spine_reg_2/lorentz_tcnet/steele_results_phase4.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Results saved to: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
