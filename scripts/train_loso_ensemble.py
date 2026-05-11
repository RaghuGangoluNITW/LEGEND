#!/usr/bin/env python3
"""
Ensemble Evaluation - Combine Baseline + Phase 3 + Phase 4
============================================================

Strategy: Use weighted voting from all three models to get best results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path

from src.lorentz_tcnet.model import TriModalLorentzNet
from src.lorentz_tcnet.model_enhanced import EnhancedTriModalLorentzNet
from src.lorentz_tcnet.model_phase4 import ImprovedTriModalLorentzNet


class TriModalDataset(Dataset):
    def __init__(self, eeg, esg, emg, labels):
        self.eeg = torch.FloatTensor(eeg)
        self.esg = torch.FloatTensor(esg)
        self.emg = torch.FloatTensor(emg)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.eeg[idx], self.esg[idx], self.emg[idx], self.labels[idx]


def load_subject_data(data_dir, subject_id):
    """Load tri-modal data for a single subject."""
    data_path = data_dir / f"{subject_id}.npz"
    data = np.load(data_path)
    return data['eeg'], data['esg'], data['emg'], data['labels']


def train_model(model, train_loader, device, epochs=50, early_stop_patience=10):
    """Train a single model with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for eeg, esg, emg, labels in train_loader:
            eeg, esg, emg, labels = eeg.to(device), esg.to(device), emg.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(eeg, esg, emg)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        
        # Early stopping based on training loss
        if avg_loss < best_loss - 0.001:
            best_loss = avg_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            if best_state is not None:
                model.load_state_dict(best_state)
            break
    
    return model


def evaluate_ensemble(subject_id, all_subjects, data_dir, device, model_weights=(0.4, 0.3, 0.3)):
    """
    Train ensemble of 3 models and evaluate with weighted voting.
    
    Args:
        model_weights: (baseline_weight, phase3_weight, phase4_weight)
    """
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
    train_dataset = TriModalDataset(train_eeg, train_esg, train_emg, train_labels)
    test_dataset = TriModalDataset(test_eeg, test_esg, test_emg, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Initialize 3 models
    print("Training Model 1 (Baseline)...")
    model1 = TriModalLorentzNet(
        eeg_channels=28, esg_channels=15, emg_channels=8,
        hidden_channels=64, latent_dim=32, num_classes=4
    ).to(device)
    model1 = train_model(model1, train_loader, device, epochs=50, early_stop_patience=10)
    
    print("Training Model 2 (Phase 3 Enhanced)...")
    model2 = EnhancedTriModalLorentzNet(
        eeg_channels=28, esg_channels=15, emg_channels=8,
        hidden_channels=64, latent_dim=32, num_classes=4,
        use_attention=True, use_spatial_attention=True
    ).to(device)
    model2 = train_model(model2, train_loader, device, epochs=50, early_stop_patience=10)
    
    print("Training Model 3 (Phase 4 Improved)...")
    model3 = ImprovedTriModalLorentzNet(
        eeg_channels=28, esg_channels=15, emg_channels=8,
        hidden_channels=64, latent_dim=32, num_classes=4,
        use_spatial_attention=True, dropout=0.2
    ).to(device)
    model3 = train_model(model3, train_loader, device, epochs=50, early_stop_patience=10)
    
    # Ensemble evaluation
    print("\nEvaluating Ensemble...")
    model1.eval()
    model2.eval()
    model3.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for eeg, esg, emg, labels in test_loader:
            eeg, esg, emg = eeg.to(device), esg.to(device), emg.to(device)
            
            # Get predictions from all 3 models
            out1 = torch.softmax(model1(eeg, esg, emg), dim=1)
            out2 = torch.softmax(model2(eeg, esg, emg), dim=1)
            out3 = torch.softmax(model3(eeg, esg, emg), dim=1)
            
            # Weighted ensemble
            w1, w2, w3 = model_weights
            ensemble_out = w1 * out1 + w2 * out2 + w3 * out3
            
            _, predicted = torch.max(ensemble_out, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = 100.0 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    print(f"\n✅ {subject_id} Ensemble Accuracy: {accuracy:.2f}%")
    
    return {
        'subject': subject_id,
        'accuracy': accuracy
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    data_dir = Path('/home/supermicro/spine_reg_2/lorentz_tcnet/data/steele')
    
    all_subjects = [
        'NIS001', 'NIS002', 'NIS003', 'NIS004', 'NIS005',
        'NIS006', 'NIS007', 'NIS008', 'NIS009', 'NIS010'
    ]
    
    print("="*80)
    print("ENSEMBLE TRAINING - Steele Dataset (LOSO)")
    print("="*80)
    print("Strategy: Weighted ensemble of 3 models")
    print("  - Model 1: Baseline (40% weight)")
    print("  - Model 2: Phase 3 Enhanced (30% weight)")
    print("  - Model 3: Phase 4 Improved (30% weight)")
    print(f"Subjects: {', '.join(all_subjects)}")
    print(f"Total: {len(all_subjects)} subjects")
    print("="*80)
    
    # LOSO with ensemble
    results = []
    for subject_id in tqdm(all_subjects, desc="LOSO Subjects"):
        result = evaluate_ensemble(subject_id, all_subjects, data_dir, device)
        results.append(result)
    
    # Statistics
    accuracies = [r['accuracy'] for r in results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)
    
    print("\n" + "="*80)
    print("FINAL RESULTS - Ensemble (Steele)")
    print("="*80)
    print()
    print(f"Mean Accuracy: {mean_acc:.2f}%")
    print(f"Std Accuracy:  {std_acc:.2f}%")
    print(f"Min Accuracy:  {min_acc:.2f}%")
    print(f"Max Accuracy:  {max_acc:.2f}%")
    print()
    
    print("COMPARISON:")
    print(f"  Baseline:       50.16% ± 13.82%")
    print(f"  Phase 3:        48.45% ± 9.64%")
    print(f"  Phase 4:        47.27% ± 10.33%")
    print(f"  Ensemble:       {mean_acc:.2f}% ± {std_acc:.2f}%")
    print()
    
    improvement = mean_acc - 50.16
    print(f"Improvement over baseline: {improvement:+.2f}%")
    
    if mean_acc > 50.16:
        print("✅ SUCCESS: Beat baseline!")
    else:
        print("❌ Still below baseline, but likely more stable")
    
    # Save results
    output_path = Path('/home/supermicro/spine_reg_2/lorentz_tcnet/steele_results_ensemble.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Results saved to: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
