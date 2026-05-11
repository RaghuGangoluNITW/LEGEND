#!/usr/bin/env python3
"""
Quick PhysioNet preprocessing and training script
Adapts Lorentzian TCN for EEG-only motor imagery dataset
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import mne
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lorentz_tcnet.model import TriModalLorentzNet


def train_epoch(model, loader, optimizer, criterion, device):
    """Training epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Training"):
        eeg = batch['eeg'].to(device)
        esg = batch['esg'].to(device)
        emg = batch['emg'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(eeg, esg, emg)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), 100.0 * correct / total


def validate_epoch(model, loader, criterion, device):
    """Validation epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            eeg = batch['eeg'].to(device)
            esg = batch['esg'].to(device)
            emg = batch['emg'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(eeg, esg, emg)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(loader), 100.0 * correct / total


class PhysioNetConfig:
    """PhysioNet configuration"""
    data_root = Path('data/physionet/files')
    cache_dir = Path('data/physionet/cached')
    
    # Use first 10 subjects for quick validation
    subjects = list(range(1, 11))  # S001 to S010
    runs = [3, 7, 11]  # Motor execution: left fist (R3), right fist (R7), both fists (R11)
    
    sample_rate = 160
    window_size = 640  # 4 seconds
    overlap = 0.5
    
    # Model params (EEG-only, so we use EEG for all three "modalities")
    eeg_channels = 64
    num_classes = 2  # Binary: left vs right fist
    
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.0005
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_physionet_run(subject_id: int, run: int, data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one PhysioNet run (EDF file with annotations)"""
    subject_str = f"S{subject_id:03d}"
    edf_file = data_root / subject_str / f"{subject_str}R{run:02d}.edf"
    
    if not edf_file.exists():
        raise FileNotFoundError(f"Missing: {edf_file}")
    
    # Load EEG data
    raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
    
    # Get annotations
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    
    # Extract data
    data = raw.get_data()  # (64 channels, time_samples)
    
    return data, events


def preprocess_physionet_subject(subject_id: int, config: PhysioNetConfig) -> Dict:
    """Preprocess one PhysioNet subject"""
    
    all_windows = []
    all_labels = []
    
    for run in config.runs:
        try:
            data, events = load_physionet_run(subject_id, run, config.data_root)
            
            # Create windows around task events
            step = int(config.window_size * (1 - config.overlap))
            
            for event_sample, _, event_id in events:
                # Skip rest events (T0)
                if event_id == 1:  # T0 = rest
                    continue
                
                # Extract window centered on event
                start = event_sample - config.window_size // 2
                end = start + config.window_size
                
                if start >= 0 and end <= data.shape[1]:
                    window = data[:, start:end]
                    
                    # Normalize
                    window = (window - window.mean(axis=1, keepdims=True)) / (window.std(axis=1, keepdims=True) + 1e-8)
                    
                    all_windows.append(window)
                    
                    # Map events to binary labels (T1=left=0, T2=right=1)
                    label = event_id - 2  # T1->0, T2->1
                    all_labels.append(label)
        
        except Exception as e:
            print(f"  Warning: Run {run} failed - {e}")
            continue
    
    if not all_windows:
        return None
    
    return {
        'eeg': np.array(all_windows, dtype=np.float32),
        'labels': np.array(all_labels, dtype=np.int64)
    }


def preprocess_all_subjects(config: PhysioNetConfig):
    """Preprocess all PhysioNet subjects"""
    
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("Preprocessing PhysioNet subjects...")
    
    for subject_id in tqdm(config.subjects, desc="Subjects"):
        cache_file = config.cache_dir / f"S{subject_id:03d}.pkl"
        
        if cache_file.exists():
            print(f"  S{subject_id:03d}: already cached")
            continue
        
        try:
            data = preprocess_physionet_subject(subject_id, config)
            
            if data is None:
                print(f"  S{subject_id:03d}: no valid data")
                continue
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"  S{subject_id:03d}: {len(data['labels'])} windows saved")
        
        except Exception as e:
            print(f"  S{subject_id:03d}: FAILED - {e}")


class PhysioNetDataset(Dataset):
    """PhysioNet dataset (EEG-only)"""
    
    def __init__(self, eeg, labels):
        self.eeg = torch.FloatTensor(eeg)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'eeg': self.eeg[idx],
            'esg': self.eeg[idx].clone(),  # Use EEG as ESG placeholder
            'emg': self.eeg[idx][:8, :],   # Use first 8 EEG channels as EMG placeholder
            'labels': self.labels[idx]
        }


def train_loso_physionet():
    """LOSO training on PhysioNet"""
    
    config = PhysioNetConfig()
    
    # Preprocess if needed
    if not config.cache_dir.exists() or len(list(config.cache_dir.glob('*.pkl'))) < len(config.subjects):
        preprocess_all_subjects(config)
    
    print(f"\n{'='*80}")
    print("PhysioNet LOSO Training (EEG-only)")
    print(f"{'='*80}\n")
    
    results = []
    
    for test_subject in config.subjects:
        print(f"\n{'='*80}")
        print(f"Test Subject: S{test_subject:03d}")
        print(f"{'='*80}\n")
        
        # Load train subjects
        train_eeg = []
        train_labels = []
        
        for subj in config.subjects:
            if subj == test_subject:
                continue
            
            cache_file = config.cache_dir / f"S{subj:03d}.pkl"
            if not cache_file.exists():
                continue
            
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            train_eeg.append(data['eeg'])
            train_labels.append(data['labels'])
        
        # Load test subject
        test_file = config.cache_dir / f"S{test_subject:03d}.pkl"
        if not test_file.exists():
            print(f"Missing test data, skipping...")
            continue
        
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f)
        
        train_eeg = np.concatenate(train_eeg, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        
        print(f"Train: {len(train_labels)} windows")
        print(f"Test:  {len(test_data['labels'])} windows\n")
        
        # Create datasets
        train_dataset = PhysioNetDataset(train_eeg, train_labels)
        test_dataset = PhysioNetDataset(test_data['eeg'], test_data['labels'])
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
        
        # Create model (adapted for EEG-only)
        model_config = {
            'eeg_channels': 64,
            'esg_channels': 64,  # Use same as EEG
            'emg_channels': 8,   # Placeholder
            'num_classes': 2,
            'latent_dim': 128,
            'tcn_filters': 64,
            'tcn_kernel_size': 5,
            'tcn_dilations': [1, 2, 4, 8, 16, 32],
            'dropout': 0.3
        }
        
        model = TriModalLorentzNet(
            eeg_channels=model_config['eeg_channels'],
            esg_channels=model_config['esg_channels'],
            emg_channels=model_config['emg_channels'],
            num_classes=model_config['num_classes'],
            latent_dim=model_config['latent_dim'],
            hidden_dim=model_config['tcn_filters'],
            dropout=model_config['dropout']
        ).to(config.device)
        
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_acc = 0.0
        
        for epoch in range(config.num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, config.device)
            test_loss, test_acc = validate_epoch(model, test_loader, criterion, config.device)
            
            if test_acc > best_acc:
                best_acc = test_acc
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:02d}: Train Acc={train_acc:.2f}% | Test Acc={test_acc:.2f}% (Best={best_acc:.2f}%)")
        
        results.append({
            'subject': f"S{test_subject:03d}",
            'accuracy': best_acc
        })
        
        print(f"\n✅ S{test_subject:03d} Best Accuracy: {best_acc:.2f}%")
    
    # Summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}\n")
    
    for res in results:
        print(f"{res['subject']}: {res['accuracy']:.2f}%")
    
    mean_acc = np.mean([r['accuracy'] for r in results])
    print(f"\n🎯 Mean LOSO Accuracy: {mean_acc:.2f}%")
    
    # Save results
    with open('physionet_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Results saved to: physionet_results.pkl")


if __name__ == '__main__':
    train_loso_physionet()
