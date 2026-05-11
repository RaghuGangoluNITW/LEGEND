#!/usr/bin/env python3
"""
BCI Competition IV-2a Training Script
9 subjects, 4-class motor imagery (left hand, right hand, feet, tongue)
22 EEG + 3 EOG channels
LOSO cross-validation for SOTA comparison
"""

import sys
import os
from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import mne
from scipy import signal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lorentz_tcnet.model import TriModalLorentzNet


class BCICompetitionDataset(Dataset):
    """BCI Competition IV-2a Dataset"""
    
    def __init__(self, windows, labels):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


def load_bci_subject(subject_id, data_dir):
    """
    Load BCI Competition IV-2a data for one subject
    
    Args:
        subject_id: Subject ID (e.g., 'A01')
        data_dir: Path to BCICIV_2a_gdf directory
    
    Returns:
        windows: (N, C, T) array
        labels: (N,) array (0-3 for 4 classes)
    """
    data_dir = Path(data_dir)
    
    # Load training data (T suffix)
    train_file = data_dir / f"{subject_id}T.gdf"
    
    if not train_file.exists():
        raise FileNotFoundError(f"File not found: {train_file}")
    
    # Read GDF file with MNE
    raw = mne.io.read_raw_gdf(str(train_file), preload=True, verbose=False)
    
    # Get EEG channels only (exclude EOG for simplicity)
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)
    raw.pick(eeg_picks)
    
    # Filter: 8-30 Hz (mu and beta bands for motor imagery)
    raw.filter(8.0, 30.0, fir_design='firwin', verbose=False)
    
    # Get events
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    
    # BCI Competition IV-2a event mapping:
    # 769: left hand, 770: right hand, 771: foot, 772: tongue
    class_mapping = {769: 0, 770: 1, 771: 2, 772: 3}
    
    # Filter events to only include motor imagery classes
    valid_events = []
    for event in events:
        if event[2] in class_mapping:
            valid_events.append(event)
    
    if len(valid_events) == 0:
        # Try alternative event codes
        class_mapping = {7: 0, 8: 1, 9: 2, 10: 3}  # Sometimes encoded differently
        for event in events:
            if event[2] in class_mapping:
                valid_events.append(event)
    
    valid_events = np.array(valid_events)
    
    if len(valid_events) == 0:
        print(f"Warning: No valid events found for {subject_id}")
        print(f"Available event codes: {np.unique(events[:, 2])}")
        # Use first 4 unique event codes as classes
        unique_codes = sorted(np.unique(events[:, 2]))[:4]
        class_mapping = {code: idx for idx, code in enumerate(unique_codes)}
        for event in events:
            if event[2] in class_mapping:
                valid_events.append(event)
        valid_events = np.array(valid_events)
    
    # Extract epochs: 0.5s to 4.5s after cue (motor imagery period)
    tmin, tmax = 0.5, 4.5
    epochs = mne.Epochs(
        raw,
        valid_events,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )
    
    # Get data
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    labels = np.array([class_mapping[event[2]] for event in valid_events if event[2] in class_mapping])
    
    # Resample to 125 Hz (standard for BCI)
    sfreq = epochs.info['sfreq']
    if sfreq != 125:
        n_times_new = int(data.shape[2] * 125 / sfreq)
        data_resampled = np.zeros((data.shape[0], data.shape[1], n_times_new))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data_resampled[i, j, :] = signal.resample(data[i, j, :], n_times_new)
        data = data_resampled
    
    # Normalize per channel
    for ch in range(data.shape[1]):
        mean = data[:, ch, :].mean()
        std = data[:, ch, :].std()
        if std > 0:
            data[:, ch, :] = (data[:, ch, :] - mean) / std
    
    return data, labels


def train_one_subject(subject_id, train_subjects, data_dir, device, num_epochs=30):
    """Train on all subjects except one (LOSO)"""
    
    print(f"\n{'='*80}")
    print(f"Test Subject: {subject_id}")
    print(f"{'='*80}")
    
    # Load all training subjects
    train_windows = []
    train_labels = []
    
    for train_subj in tqdm(train_subjects, desc="Loading train subjects"):
        try:
            windows, labels = load_bci_subject(train_subj, data_dir)
            train_windows.append(windows)
            train_labels.append(labels)
        except Exception as e:
            print(f"Error loading {train_subj}: {e}")
            continue
    
    # Load test subject
    try:
        test_windows, test_labels = load_bci_subject(subject_id, data_dir)
    except Exception as e:
        print(f"Error loading test subject {subject_id}: {e}")
        return None
    
    # Concatenate training data
    train_windows = np.concatenate(train_windows, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    
    print(f"\nTrain: {len(train_labels)} trials")
    print(f"Test:  {len(test_labels)} trials")
    print(f"Channels: {train_windows.shape[1]}, Time points: {train_windows.shape[2]}")
    
    # Carve out val split (20%) from training data, stratified
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(train_labels))
    idx_tr, idx_val = train_test_split(idx, test_size=0.20, random_state=42, stratify=train_labels)
    val_windows = train_windows[idx_val]
    val_labels  = train_labels[idx_val]
    train_windows = train_windows[idx_tr]
    train_labels  = train_labels[idx_tr]

    # Create datasets
    train_dataset = BCICompetitionDataset(train_windows, train_labels)
    val_dataset   = BCICompetitionDataset(val_windows,   val_labels)
    test_dataset  = BCICompetitionDataset(test_windows,  test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)
    
    # Model configuration for EEG-only, 4-class
    n_channels = train_windows.shape[1]
    n_timepoints = train_windows.shape[2]
    n_classes = 4
    
    model = TriModalLorentzNet(
        eeg_channels=n_channels,
        esg_channels=0,  # No ESG for BCI dataset
        emg_channels=0,  # No EMG for BCI dataset
        hidden_dim=64,
        latent_dim=32,
        num_classes=n_classes,
        dropout=0.5
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x, None, None)  # EEG-only
            logits = outputs['logits']
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            _, predicted = logits.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

        train_acc = 100.0 * train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x, None, None)
                _, predicted = outputs['logits'].max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        val_acc = 100.0 * val_correct / val_total

        # Test (for logging only — NOT used for model selection)
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x, None, None)
                _, predicted = outputs['logits'].max(1)
                test_total += batch_y.size(0)
                test_correct += predicted.eq(batch_y).sum().item()
        test_acc = 100.0 * test_correct / test_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step(val_acc)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d}: Train={train_acc:.2f}% | Val={val_acc:.2f}% | Test={test_acc:.2f}% (BestVal={best_val_acc:.2f}%)")

    print(f"\n✅ {subject_id}  test@best-val: {best_test_acc:.2f}%  (best val: {best_val_acc:.2f}%)")

    return {
        'subject': subject_id,
        'accuracy': best_test_acc,
        'best_val_acc': best_val_acc,
        'n_train': len(train_labels),
        'n_test': len(test_labels)
    }


def main():
    print("="*80)
    print("BCI Competition IV-2a LOSO Training")
    print("="*80)
    
    # Configuration
    data_dir = Path(__file__).parent.parent / 'data' / 'bciciv_2a' / 'BCICIV_2a_gdf'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    
    # All subjects
    all_subjects = [f'A{i:02d}' for i in range(1, 10)]  # A01 to A09
    
    print(f"\nSubjects: {', '.join(all_subjects)}")
    print(f"Total: {len(all_subjects)} subjects")
    print("="*80)
    
    # LOSO training
    results = []
    
    for subject_id in tqdm(all_subjects, desc="LOSO Subjects"):
        train_subjects = [s for s in all_subjects if s != subject_id]
        
        result = train_one_subject(
            subject_id=subject_id,
            train_subjects=train_subjects,
            data_dir=data_dir,
            device=device,
            num_epochs=30
        )
        
        if result is not None:
            results.append(result)
    
    # Compute statistics
    accuracies = [r['accuracy'] for r in results]
    
    print("\n" + "="*80)
    print("FINAL RESULTS - BCI Competition IV-2a (9 subjects)")
    print("="*80)
    print(f"\nMean Accuracy: {np.mean(accuracies):.2f}%")
    print(f"Std Accuracy:  {np.std(accuracies):.2f}%")
    print(f"Min Accuracy:  {np.min(accuracies):.2f}%")
    print(f"Max Accuracy:  {np.max(accuracies):.2f}%")
    
    # Save results
    output_file = Path(__file__).parent.parent / 'bciciv2a_results_final.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("="*80)


if __name__ == '__main__':
    main()
