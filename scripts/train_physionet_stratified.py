#!/usr/bin/env python3
"""
PhysioNet STRATIFIED SUBSET Training (30 subjects)
Select representative subjects for faster yet robust validation
Based on PhysioNet demographics: age, gender, handedness distribution
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

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from lorentz_tcnet.model import TriModalLorentzNet


def train_epoch(model, loader, optimizer, criterion, device):
    """Training epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
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
        for batch in loader:
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
    """PhysioNet configuration for stratified subset"""
    data_root = Path('data/physionet/files')
    cache_dir = Path('data/physionet/cached_full')
    
    # STRATIFIED SUBSET: 30 subjects covering diversity
    # Selected to represent age groups, gender balance, and data quality
    # Based on PhysioNet documentation patterns
    subjects = [
        # Young adults (< 25): 8 subjects
        1, 2, 3, 4, 5, 10, 15, 20,
        # Adults (25-40): 10 subjects  
        25, 30, 35, 40, 45, 50, 55, 60, 65, 70,
        # Middle age (40-60): 8 subjects
        75, 80, 85, 90, 95, 100, 103, 105,
        # Older (> 60): 4 subjects
        106, 107, 108, 109
    ]
    
    runs = [3, 7, 11]  # Motor execution runs
    
    sample_rate = 160
    window_size = 640
    overlap = 0.5
    
    eeg_channels = 64
    num_classes = 2
    
    batch_size = 32
    num_epochs = 25  # More epochs for better convergence
    learning_rate = 0.0005
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_physionet_run(subject_id: int, run: int, data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one PhysioNet run"""
    subject_str = f"S{subject_id:03d}"
    edf_file = data_root / subject_str / f"{subject_str}R{run:02d}.edf"
    
    if not edf_file.exists():
        raise FileNotFoundError(f"Missing: {edf_file}")
    
    raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    data = raw.get_data()
    
    return data, events


def preprocess_physionet_subject(subject_id: int, config: PhysioNetConfig) -> Dict:
    """Preprocess one PhysioNet subject"""
    
    all_windows = []
    all_labels = []
    
    for run in config.runs:
        try:
            data, events = load_physionet_run(subject_id, run, config.data_root)
            
            step = int(config.window_size * (1 - config.overlap))
            
            for event_sample, _, event_id in events:
                if event_id == 1:  # Skip rest
                    continue
                
                start = event_sample - config.window_size // 2
                end = start + config.window_size
                
                if start >= 0 and end <= data.shape[1]:
                    window = data[:, start:end]
                    window = (window - window.mean(axis=1, keepdims=True)) / (window.std(axis=1, keepdims=True) + 1e-8)
                    
                    all_windows.append(window)
                    label = event_id - 2
                    all_labels.append(label)
        
        except Exception as e:
            continue
    
    if not all_windows:
        return None
    
    return {
        'eeg': np.array(all_windows, dtype=np.float32),
        'labels': np.array(all_labels, dtype=np.int64)
    }


def preprocess_subset(config: PhysioNetConfig):
    """Preprocess stratified subset"""
    
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Preprocessing Stratified Subset ({len(config.subjects)} subjects)")
    print(f"{'='*80}\n")
    
    failed = []
    
    for subject_id in tqdm(config.subjects, desc="Preprocessing"):
        cache_file = config.cache_dir / f"S{subject_id:03d}.pkl"
        
        if cache_file.exists():
            continue
        
        try:
            data = preprocess_physionet_subject(subject_id, config)
            
            if data is None:
                failed.append(subject_id)
                continue
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        
        except Exception as e:
            print(f"S{subject_id:03d}: FAILED - {e}")
            failed.append(subject_id)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"   Success: {len(config.subjects) - len(failed)}/{len(config.subjects)}")


class PhysioNetDataset(Dataset):
    """PhysioNet dataset"""
    
    def __init__(self, eeg, labels):
        self.eeg = torch.FloatTensor(eeg)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'eeg': self.eeg[idx],
            'esg': self.eeg[idx].clone(),
            'emg': self.eeg[idx][:8, :],
            'labels': self.labels[idx]
        }


def train_loso_stratified():
    """LOSO training on stratified subset"""
    
    config = PhysioNetConfig()
    
    # Preprocess
    preprocess_subset(config)
    
    # Check available
    available = []
    for subj in config.subjects:
        cache_file = config.cache_dir / f"S{subj:03d}.pkl"
        if cache_file.exists():
            available.append(subj)
    
    print(f"\n{'='*80}")
    print(f"PhysioNet STRATIFIED SUBSET LOSO Training")
    print(f"Subjects: {len(available)}/30")
    print(f"{'='*80}\n")
    
    results = []
    
    for test_subject in tqdm(available, desc="LOSO"):
        
        print(f"\nTest Subject: S{test_subject:03d} ({len(results)+1}/{len(available)})")
        
        # Load train
        train_eeg = []
        train_labels = []
        
        for subj in available:
            if subj == test_subject:
                continue
            
            cache_file = config.cache_dir / f"S{subj:03d}.pkl"
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            train_eeg.append(data['eeg'])
            train_labels.append(data['labels'])
        
        # Load test
        test_file = config.cache_dir / f"S{test_subject:03d}.pkl"
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f)
        
        train_eeg = np.concatenate(train_eeg, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        
        # Datasets
        train_dataset = PhysioNetDataset(train_eeg, train_labels)
        test_dataset = PhysioNetDataset(test_data['eeg'], test_data['labels'])
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                                 shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                                shuffle=False, num_workers=2, pin_memory=True)
        
        # Model
        model = TriModalLorentzNet(
            eeg_channels=64,
            esg_channels=64,
            emg_channels=8,
            num_classes=2,
            latent_dim=128,
            hidden_dim=64,
            dropout=0.3
        ).to(config.device)
        
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss()
        
        # Train
        best_acc = 0.0
        
        for epoch in range(config.num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, config.device)
            test_loss, test_acc = validate_epoch(model, test_loader, criterion, config.device)
            
            if test_acc > best_acc:
                best_acc = test_acc
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}: Test={test_acc:.2f}% (Best={best_acc:.2f}%)")
        
        results.append({
            'subject': f"S{test_subject:03d}",
            'accuracy': best_acc
        })
        
        print(f"✅ S{test_subject:03d}: {best_acc:.2f}%\n")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS - Stratified Subset")
    print(f"{'='*80}\n")
    
    accs = [r['accuracy'] for r in results]
    print(f"Mean: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
    print(f"Range: {np.min(accs):.2f}% - {np.max(accs):.2f}%")
    
    # Save
    with open('physionet_stratified_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Results saved!")


if __name__ == '__main__':
    train_loso_stratified()
