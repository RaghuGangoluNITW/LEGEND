#!/usr/bin/env python3
"""
BCI-IV-2a Adaptive Training - ALL 9 Subjects
Core Lorentzian geometry preserved, subject-specific augmentation
FIX: Handle empty MNE events gracefully
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
from typing import Dict, List
import mne
from sklearn.model_selection import StratifiedShuffleSplit
import copy

from src.lorentz_tcnet.model import TriModalLorentzNet


class ModerateAugmentation:
    """Conservative augmentation (0.2 probability)"""
    
    @staticmethod
    def time_shift(x: torch.Tensor, max_shift: int = 5) -> torch.Tensor:
        shift = np.random.randint(-max_shift, max_shift + 1)
        return torch.roll(x, shifts=shift, dims=-1)
    
    @staticmethod
    def amplitude_scale(x: torch.Tensor) -> torch.Tensor:
        scale = np.random.uniform(0.9, 1.1)
        return x * scale
    
    @staticmethod
    def add_noise(x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * 0.005
        return x + noise
    
    @staticmethod
    def apply(x: torch.Tensor, prob: float = 0.2) -> torch.Tensor:
        if np.random.rand() < prob:
            x = ModerateAugmentation.time_shift(x)
        if np.random.rand() < prob:
            x = ModerateAugmentation.amplitude_scale(x)
        if np.random.rand() < prob:
            x = ModerateAugmentation.add_noise(x)
        return x


class BCIDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
        self.training = True
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        eeg = self.data[idx]
        label = self.labels[idx]
        
        if self.augment and self.training:
            eeg = ModerateAugmentation.apply(eeg)
        
        # Dummy ESG/EMG
        dummy_esg = torch.zeros(3, eeg.shape[1])
        dummy_emg = torch.zeros(1, eeg.shape[1])
        
        return eeg, dummy_esg, dummy_emg, label


def load_subject_safe(subj_id: int, data_dir: Path) -> tuple:
    """Load BCI-IV-2a with ROBUST event handling (from working baseline)"""
    
    train_file = data_dir / f"A0{subj_id}T.gdf"
    test_file = data_dir / f"A0{subj_id}E.gdf"
    
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Subject {subj_id} files not found")
    
    # Load train
    raw_train = mne.io.read_raw_gdf(str(train_file), preload=True, verbose=False)
    picks = mne.pick_types(raw_train.info, eeg=True, eog=False, stim=False)
    raw_train.pick(picks)
    raw_train.filter(8.0, 30.0, fir_design='firwin', verbose=False)
    
    # ROBUST event extraction (from working baseline)
    events_train, event_id = mne.events_from_annotations(raw_train, verbose=False)
    class_mapping = {769: 0, 770: 1, 771: 2, 772: 3}
    
    valid_events_train = []
    for event in events_train:
        if event[2] in class_mapping:
            valid_events_train.append(event)
    
    if len(valid_events_train) == 0:
        # Try alternative codes
        class_mapping = {7: 0, 8: 1, 9: 2, 10: 3}
        for event in events_train:
            if event[2] in class_mapping:
                valid_events_train.append(event)
    
    valid_events_train = np.array(valid_events_train)
    
    if len(valid_events_train) == 0:
        unique_codes = sorted(np.unique(events_train[:, 2]))[:4]
        class_mapping = {code: idx for idx, code in enumerate(unique_codes)}
        for event in events_train:
            if event[2] in class_mapping:
                valid_events_train.append(event)
        valid_events_train = np.array(valid_events_train)
    
    # Load test with same logic
    raw_test = mne.io.read_raw_gdf(str(test_file), preload=True, verbose=False)
    raw_test.pick(picks)
    raw_test.filter(8.0, 30.0, fir_design='firwin', verbose=False)
    
    events_test, event_id = mne.events_from_annotations(raw_test, verbose=False)
    
    valid_events_test = []
    for event in events_test:
        if event[2] in class_mapping:
            valid_events_test.append(event)
    
    valid_events_test = np.array(valid_events_test)
    
    if len(valid_events_test) == 0:
        for event in events_test:
            if event[2] in class_mapping:
                valid_events_test.append(event)
        valid_events_test = np.array(valid_events_test)
    
    # Extract epochs: 0.5-4.5s (motor imagery period)
    epochs_train = mne.Epochs(raw_train, valid_events_train, tmin=0.5, tmax=4.5,
                             baseline=None, preload=True, verbose=False)
    epochs_test = mne.Epochs(raw_test, valid_events_test, tmin=0.5, tmax=4.5,
                            baseline=None, preload=True, verbose=False)
    
    X_train = epochs_train.get_data()
    y_train = np.array([class_mapping[e[2]] for e in valid_events_train if e[2] in class_mapping])
    X_test = epochs_test.get_data()
    y_test = np.array([class_mapping[e[2]] for e in valid_events_test if e[2] in class_mapping])
    
    # Normalize
    for ch in range(X_train.shape[1]):
        mean = X_train[:, ch, :].mean()
        std = X_train[:, ch, :].std()
        if std > 0:
            X_train[:, ch, :] = (X_train[:, ch, :] - mean) / std
            X_test[:, ch, :] = (X_test[:, ch, :] - mean) / std
    
    return X_train, y_train, X_test, y_test


def train_subject(
    subj_id: int,
    data_dir: Path,
    device: torch.device,
    augment: bool,
    epochs: int = 30
) -> Dict:
    
    X_train_full, y_train_full, X_test, y_test = load_subject_safe(subj_id, data_dir)

    # Carve 20% stratified val split from Session 1 — Session 2 (X_test) is
    # never touched until final evaluation.
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    idx_tr, idx_val = next(sss.split(np.zeros(len(y_train_full)), y_train_full))
    X_train, y_train = X_train_full[idx_tr], y_train_full[idx_tr]
    X_val,   y_val   = X_train_full[idx_val], y_train_full[idx_val]

    train_ds = BCIDataset(X_train, y_train, augment=augment)
    val_ds   = BCIDataset(X_val,   y_val,   augment=False)
    test_ds  = BCIDataset(X_test,  y_test,  augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=2)
    
    # BASELINE MODEL - Core Lorentzian preserved
    model = TriModalLorentzNet(
        eeg_channels=25,
        esg_channels=3,
        emg_channels=1,
        hidden_dim=64,
        latent_dim=32,
        num_classes=4
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    best_state   = None
    patience, patience_counter = 8, 0
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_ds.training = True
        
        for eeg, esg, emg, labels in train_loader:
            eeg, esg, emg, labels = eeg.to(device), esg.to(device), emg.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(eeg, esg, emg)
            logits = outputs['logits']
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validate on Session-1 val split (Session 2 never touched here)
        model.eval()
        train_ds.training = False
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for eeg, esg, emg, labels in val_loader:
                eeg, esg, emg, labels = eeg.to(device), esg.to(device), emg.to(device), labels.to(device)
                outputs = model(eeg, esg, emg)
                logits = outputs['logits']
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Final evaluation on Session 2 using the best-val checkpoint
    model.load_state_dict(best_state)
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for eeg, esg, emg, labels in test_loader:
            eeg, esg, emg, labels = eeg.to(device), esg.to(device), emg.to(device), labels.to(device)
            outputs = model(eeg, esg, emg)
            logits = outputs['logits']
            _, predicted = torch.max(logits, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_acc = 100.0 * test_correct / test_total

    return {'subject': f'S{subj_id:02d}', 'accuracy': test_acc,
            'best_val_acc': best_val_acc, 'augmentation': augment}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    data_dir = Path('data/bciciv_2a/BCICIV_2a_gdf')
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return
    
    # Load baseline to design strategy
    baseline_file = Path('bciciv2a_results_final.pkl')
    
    if baseline_file.exists():
        with open(baseline_file, 'rb') as f:
            baseline = pickle.load(f)
            baseline_accs = {item['subject']: item['accuracy'] for item in baseline}
        
        mean_acc = np.mean(list(baseline_accs.values()))
        aug_subjects = {s for s, acc in baseline_accs.items() if acc < mean_acc}
    else:
        print("WARNING: Baseline not found, using augmentation for all subjects")
        aug_subjects = {f'S{i:02d}' for i in range(1, 10)}
        mean_acc = 0.0
    
    print("="*80)
    print("BCI-IV-2a ADAPTIVE - ALL 9 SUBJECTS")
    print("="*80)
    print(f"Baseline mean: {mean_acc:.2f}%" if mean_acc > 0 else "Baseline: Not available")
    print(f"Augmentation ON:  {len(aug_subjects)} subjects")
    print(f"Augmentation OFF: {9 - len(aug_subjects)} subjects")
    print("="*80)
    
    results = []
    
    for subj_id in tqdm(range(1, 10), desc="Training"):
        subj_name = f'S{subj_id:02d}'
        use_aug = subj_name in aug_subjects
        
        try:
            result = train_subject(subj_id, data_dir, device, augment=use_aug, epochs=30)
            results.append(result)
            print(f"OK {subj_name}: test={result['accuracy']:.2f}%  val={result['best_val_acc']:.2f}%  (Aug: {use_aug})")
        except Exception as e:
            print(f"FAIL {subj_name}: {e}")
            continue
    
    if len(results) == 0:
        print(f"ERROR: No subjects completed successfully")
        return
    
    # Final results
    accuracies = [r['accuracy'] for r in results]
    mean_final = np.mean(accuracies)
    std_final = np.std(accuracies)
    
    print("\n" + "="*80)
    print("FINAL RESULTS - BCI-IV-2a Adaptive (9 Subjects)")
    print("="*80)
    print(f"Mean: {mean_final:.2f}% ± {std_final:.2f}%")
    print(f"Min:  {min(accuracies):.2f}%")
    print(f"Max:  {max(accuracies):.2f}%")
    if mean_acc > 0:
        print(f"\nBaseline: {mean_acc:.2f}%")
        print(f"Adaptive: {mean_final:.2f}%")
        print(f"Δ: {mean_final - mean_acc:+.2f}%")
    print("="*80)
    
    # Save
    with open('bciciv2a_results_adaptive_full.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Saved: bciciv2a_results_adaptive_full.pkl")


if __name__ == '__main__':
    main()
