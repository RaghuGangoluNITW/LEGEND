#!/usr/bin/env python3
"""
Phase 3 Enhanced Training - BCI Competition IV-2a
==================================================

Enhanced model with spatial-temporal attention and data augmentation.
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
import mne

from src.lorentz_tcnet.model_enhanced import EnhancedTriModalLorentzNet, TimeSeriesAugmentation


class BCICompetitionDataset(Dataset):
    def __init__(self, data, labels, augment=False):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        
        if self.augment:
            x = TimeSeriesAugmentation.apply_augmentations(x.unsqueeze(0), aug_prob=0.5).squeeze(0)
        
        return x, self.labels[idx]


def load_bci_subject(data_dir, subject_id):
    """Load BCI Competition IV-2a data for one subject."""
    train_file = data_dir / f"{subject_id}T.gdf"
    eval_file = data_dir / f"{subject_id}E.gdf"
    
    # Load training session
    raw_train = mne.io.read_raw_gdf(train_file, preload=True, verbose=False)
    events_train, event_id_train = mne.events_from_annotations(raw_train, verbose=False)
    
    # Load evaluation session
    raw_eval = mne.io.read_raw_gdf(eval_file, preload=True, verbose=False)
    events_eval, event_id_eval = mne.events_from_annotations(raw_eval, verbose=False)
    
    # Extract EEG channels (first 22 channels)
    eeg_channels = raw_train.ch_names[:22]
    raw_train.pick_channels(eeg_channels)
    raw_eval.pick_channels(eeg_channels)
    
    # Filter: mu and beta bands (8-30 Hz)
    raw_train.filter(8, 30, fir_design='firwin', verbose=False)
    raw_eval.filter(8, 30, fir_design='firwin', verbose=False)
    
    # Event mapping: 769=left, 770=right, 771=foot, 772=tongue
    event_mapping = {769: 0, 770: 1, 771: 2, 772: 3}
    
    # Extract epochs (0.5-4.5s motor imagery period)
    def extract_epochs(raw, events, event_id):
        epochs_list = []
        labels_list = []
        
        for event_code in [769, 770, 771, 772]:
            if event_code in event_id.values():
                epochs = mne.Epochs(
                    raw, events, event_id={str(event_code): event_code},
                    tmin=0.5, tmax=4.5, baseline=None, preload=True, verbose=False
                )
                data = epochs.get_data()  # (n_trials, n_channels, n_times)
                epochs_list.append(data)
                labels_list.extend([event_mapping[event_code]] * len(data))
        
        return np.concatenate(epochs_list, axis=0), np.array(labels_list)
    
    train_data, train_labels = extract_epochs(raw_train, events_train, event_id_train)
    eval_data, eval_labels = extract_epochs(raw_eval, events_eval, event_id_eval)
    
    # Combine training and evaluation sessions
    all_data = np.concatenate([train_data, eval_data], axis=0)
    all_labels = np.concatenate([train_labels, eval_labels], axis=0)
    
    # Resample to 125 Hz
    n_samples_target = 500  # 4s at 125 Hz
    from scipy.signal import resample
    all_data_resampled = resample(all_data, n_samples_target, axis=2)
    
    # Normalize per-channel
    mean = all_data_resampled.mean(axis=(0, 2), keepdims=True)
    std = all_data_resampled.std(axis=(0, 2), keepdims=True)
    all_data_resampled = (all_data_resampled - mean) / (std + 1e-6)
    
    return all_data_resampled, all_labels


def train_one_subject(subject_id, all_subjects, data_dir, device, num_epochs=30):
    """LOSO training for one subject."""
    
    print(f"\n{'='*80}")
    print(f"Test Subject: {subject_id}")
    print('='*80)
    
    # Split
    train_subjects = [s for s in all_subjects if s != subject_id]
    
    # Load training data
    train_windows, train_labels = [], []
    print("Loading train subjects:", end=" ", flush=True)
    for subj in tqdm(train_subjects, desc="Loading train subjects"):
        data, labels = load_bci_subject(data_dir, subj)
        train_windows.append(data)
        train_labels.append(labels)
    
    train_windows = np.concatenate(train_windows, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    
    # Load test data
    test_windows, test_labels = load_bci_subject(data_dir, subject_id)
    
    n_train_total = len(train_labels)
    n_test = len(test_labels)
    n_channels = train_windows.shape[1]
    n_timepoints = train_windows.shape[2]
    
    print(f"\nTrain: {n_train_total} trials")
    print(f"Test:  {n_test} trials")
    print(f"Channels: {n_channels}, Time points: {n_timepoints}")
    
    # Datasets with augmentation
    train_dataset = BCICompetitionDataset(train_windows, train_labels, augment=True)
    test_dataset = BCICompetitionDataset(test_windows, test_labels, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Phase 3 Enhanced Model
    model = EnhancedTriModalLorentzNet(
        eeg_channels=n_channels,
        esg_channels=0,
        emg_channels=0,
        hidden_dim=64,
        latent_dim=32,
        num_classes=4,
        dropout=0.5,
        use_attention=True
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    best_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x, None, None)
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
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x, None, None)
                logits = outputs['logits']
                _, predicted = logits.max(1)
                test_total += batch_y.size(0)
                test_correct += predicted.eq(batch_y).sum().item()
        
        test_acc = 100.0 * test_correct / test_total
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        scheduler.step(test_acc)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d}: Train={train_acc:.2f}% | Test={test_acc:.2f}% (Best={best_acc:.2f}%)")
    
    print(f"\n✅ {subject_id} Best: {best_acc:.2f}%")
    
    return {
        'subject': subject_id,
        'accuracy': best_acc,
        'n_train': n_train_total,
        'n_test': n_test
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*80)
    print("Phase 3 Enhanced Training - BCI Competition IV-2a (LOSO)")
    print("="*80)
    print(f"Device: {device}")
    
    data_dir = Path(__file__).parent.parent / 'data' / 'bciciv_2a' / 'BCICIV_2a_gdf'
    
    all_subjects = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']
    
    print(f"\nSubjects: {', '.join(all_subjects)}")
    print(f"Total: {len(all_subjects)} subjects")
    print("="*80)
    
    results = []
    
    for subject_id in tqdm(all_subjects, desc="LOSO Subjects"):
        result = train_one_subject(
            subject_id=subject_id,
            all_subjects=all_subjects,
            data_dir=data_dir,
            device=device,
            num_epochs=30
        )
        results.append(result)
    
    # Statistics
    accuracies = [r['accuracy'] for r in results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)
    
    print("\n" + "="*80)
    print("FINAL RESULTS - BCI Competition IV-2a (Phase 3 Enhanced)")
    print("="*80)
    print(f"\nMean Accuracy: {mean_acc:.2f}%")
    print(f"Std Accuracy:  {std_acc:.2f}%")
    print(f"Min Accuracy:  {min_acc:.2f}%")
    print(f"Max Accuracy:  {max_acc:.2f}%")
    
    # Save
    output_path = Path(__file__).parent.parent / 'bciciv2a_results_enhanced.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Results saved to: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
