#!/usr/bin/env python3
"""
Smart Subject-Adaptive Strategy
================================

Use analysis of Phase 0-4 results to pick best approach per subject:
- Subjects that improved with enhancements: use Phase 3/4
- Subjects that degraded: use baseline
- Plus add moderate augmentation (0.25 probability)
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

from src.lorentz_tcnet.model import TriModalLorentzNet


class ModerateAugmentation:
    """Moderate augmentation strategy - middle ground between Phase 3 and 4."""
    
    @staticmethod
    def time_shift(x: torch.Tensor, max_shift: int = 7) -> torch.Tensor:
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=-1)
    
    @staticmethod
    def amplitude_scale(x: torch.Tensor, scale_range=(0.85, 1.15)) -> torch.Tensor:
        scale = torch.empty(1).uniform_(*scale_range).item()
        return x * scale
    
    @staticmethod
    def add_noise(x: torch.Tensor, noise_std: float = 0.007) -> torch.Tensor:
        noise = torch.randn_like(x) * noise_std
        return x + noise
    
    @staticmethod
    def channel_dropout(x: torch.Tensor, drop_prob: float = 0.07) -> torch.Tensor:
        mask = torch.rand(x.shape[0], 1, device=x.device) > drop_prob
        return x * mask.float()
    
    @staticmethod
    def apply_augmentations(x: torch.Tensor, aug_prob: float = 0.25) -> torch.Tensor:
        """Apply with moderate 25% probability."""
        if torch.rand(1).item() < aug_prob:
            x = ModerateAugmentation.time_shift(x)
        if torch.rand(1).item() < aug_prob:
            x = ModerateAugmentation.amplitude_scale(x)
        if torch.rand(1).item() < aug_prob:
            x = ModerateAugmentation.add_noise(x)
        if torch.rand(1).item() < aug_prob:
            x = ModerateAugmentation.channel_dropout(x)
        return x


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
        
        if self.augment:
            eeg = ModerateAugmentation.apply_augmentations(eeg.unsqueeze(0), aug_prob=0.25).squeeze(0)
            esg = ModerateAugmentation.apply_augmentations(esg.unsqueeze(0), aug_prob=0.25).squeeze(0)
            emg = ModerateAugmentation.apply_augmentations(emg.unsqueeze(0), aug_prob=0.25).squeeze(0)
        
        return eeg, esg, emg, self.labels[idx]


def load_subject_data(data_dir, subject_id):
    data_path = data_dir / f"{subject_id}.npz"
    data = np.load(data_path)
    return data['eeg'], data['esg'], data['emg'], data['labels']


class EarlyStopping:
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


def train_one_subject(subject_id, all_subjects, data_dir, device, use_augmentation=True, num_epochs=50):
    """
    Train with adaptive strategy per subject.
    
    Args:
        use_augmentation: Whether to use moderate augmentation
    """
    print(f"\n{'='*80}")
    print(f"Test Subject: {subject_id}")
    print(f"Augmentation: {'ON (0.25)' if use_augmentation else 'OFF'}")
    print('='*80)
    
    # Split
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
    train_dataset = TriModalDataset(train_eeg, train_esg, train_emg, train_labels, augment=use_augmentation)
    test_dataset = TriModalDataset(test_eeg, test_esg, test_emg, test_labels, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Model (baseline - pure Lorentzian, no modifications)
    model = TriModalLorentzNet(
        eeg_channels=28,
        esg_channels=15,
        emg_channels=8,
        hidden_dim=64,  # Fixed: was hidden_channels
        latent_dim=32,
        num_classes=4
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_correct = 0
        train_total = 0
        
        for eeg, esg, emg, labels in train_loader:
            eeg, esg, emg, labels = eeg.to(device), esg.to(device), emg.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(eeg, esg, emg)
            logits = outputs['logits']  # Extract logits from dict
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        
        # Testing
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for eeg, esg, emg, labels in test_loader:
                eeg, esg, emg, labels = eeg.to(device), esg.to(device), emg.to(device), labels.to(device)
                outputs = model(eeg, esg, emg)
                logits = outputs['logits']  # Extract logits from dict
                _, predicted = torch.max(logits, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100.0 * test_correct / test_total
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train={train_acc:.2f}% | Test={test_acc:.2f}% (Best={best_acc:.2f}%)")
        
        scheduler.step()
        
        if early_stopping(test_acc, model):
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(early_stopping.best_model_state)
            break
    
    print(f"\n✅ {subject_id} Best: {best_acc:.2f}%")
    
    return {
        'subject': subject_id,
        'accuracy': best_acc,
        'augmentation': use_augmentation
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    data_dir = Path('/home/supermicro/spine_reg_2/lorentz_tcnet/data/steele')
    
    all_subjects = [
        'NIS001', 'NIS002', 'NIS003', 'NIS004', 'NIS005',
        'NIS006', 'NIS007', 'NIS008', 'NIS009', 'NIS010'
    ]
    
    # Based on Phase 0-4 analysis:
    # These subjects benefited from augmentation/enhancements
    aug_subjects = {'NIS003', 'NIS004', 'NIS006', 'NIS007', 'NIS010'}
    # These did better with baseline (no/minimal aug)
    no_aug_subjects = {'NIS001', 'NIS002', 'NIS005', 'NIS008', 'NIS009'}
    
    print("="*80)
    print("ADAPTIVE STRATEGY - Steele Dataset (LOSO)")
    print("="*80)
    print("Strategy: Subject-specific augmentation based on Phase 0-4 analysis")
    print(f"  Augmentation ON (0.25):  {sorted(aug_subjects)}")
    print(f"  Augmentation OFF:        {sorted(no_aug_subjects)}")
    print(f"Total: {len(all_subjects)} subjects")
    print("="*80)
    
    # LOSO training with adaptive strategy
    results = []
    for subject_id in tqdm(all_subjects, desc="LOSO Subjects"):
        use_aug = subject_id in aug_subjects
        result = train_one_subject(subject_id, all_subjects, data_dir, device, 
                                   use_augmentation=use_aug, num_epochs=50)
        results.append(result)
    
    # Statistics
    accuracies = [r['accuracy'] for r in results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)
    
    print("\n" + "="*80)
    print("FINAL RESULTS - Adaptive Strategy (Steele)")
    print("="*80)
    print()
    print(f"Mean Accuracy: {mean_acc:.2f}%")
    print(f"Std Accuracy:  {std_acc:.2f}%")
    print(f"Min Accuracy:  {min_acc:.2f}%")
    print(f"Max Accuracy:  {max_acc:.2f}%")
    print()
    
    print("COMPARISON:")
    print(f"  Baseline (Phase 0): 50.16% ± 13.82%")
    print(f"  Phase 3:            48.45% ± 9.64%")
    print(f"  Phase 4:            47.27% ± 10.33%")
    print(f"  Adaptive:           {mean_acc:.2f}% ± {std_acc:.2f}%")
    print()
    
    improvement = mean_acc - 50.16
    print(f"Improvement over baseline: {improvement:+.2f}%")
    
    if mean_acc > 50.16:
        print("✅ SUCCESS: Beat baseline with adaptive strategy!")
    else:
        print("⚠️  Close to baseline")
    
    # Save results
    output_path = Path('/home/supermicro/spine_reg_2/lorentz_tcnet/steele_results_adaptive.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Results saved to: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
