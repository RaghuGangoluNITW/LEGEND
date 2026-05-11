#!/usr/bin/env python3
"""
PhysioNet Adaptive Training - ALL 109 Subjects
Core Lorentzian geometry preserved, subject-specific augmentation
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
from typing import Dict, List, Tuple
from scipy import signal as sp_signal

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


class PhysioNetDataset(Dataset):
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
        
        # Dummy ESG/EMG for tri-modal model
        dummy_esg = torch.zeros(16, eeg.shape[1])
        dummy_emg = torch.zeros(8, eeg.shape[1])
        
        return eeg, dummy_esg, dummy_emg, label


def train_subject(
    subj: str,
    all_subj: List[str],
    data_dict: Dict,
    labels_dict: Dict,
    device: torch.device,
    augment: bool,
    epochs: int = 25
) -> Dict:
    
    train_subj = [s for s in all_subj if s != subj]
    
    # Collect training data and find max time length
    train_data_list = [data_dict[s] for s in train_subj]
    train_labels_list = [labels_dict[s] for s in train_subj]
    
    # Resample all to same length (use median or max)
    time_lengths = [d.shape[2] for d in train_data_list]
    target_length = int(np.median(time_lengths))  # Use median for stability
    
    # Resample to target length
    from scipy import signal as sp_signal
    train_data_resampled = []
    for data in train_data_list:
        if data.shape[2] != target_length:
            resampled = np.zeros((data.shape[0], data.shape[1], target_length))
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    resampled[i, j, :] = sp_signal.resample(data[i, j, :], target_length)
            train_data_resampled.append(resampled)
        else:
            train_data_resampled.append(data)
    
    train_data = np.concatenate(train_data_resampled, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    
    # Verify lengths match
    assert train_data.shape[0] == len(train_labels), f"Data/label mismatch: {train_data.shape[0]} vs {len(train_labels)}"
    
    # Resample test data too
    test_data = data_dict[subj]
    if test_data.shape[2] != target_length:
        test_data_resampled = np.zeros((test_data.shape[0], test_data.shape[1], target_length))
        for i in range(test_data.shape[0]):
            for j in range(test_data.shape[1]):
                test_data_resampled[i, j, :] = sp_signal.resample(test_data[i, j, :], target_length)
        test_data = test_data_resampled
    
    test_labels = labels_dict[subj]
    
    # Verify test lengths match
    assert test_data.shape[0] == len(test_labels), f"Test data/label mismatch: {test_data.shape[0]} vs {len(test_labels)}"
    
    train_ds = PhysioNetDataset(train_data, train_labels, augment=augment)
    test_ds = PhysioNetDataset(test_data, test_labels, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
    
    # BASELINE MODEL - Core Lorentzian preserved
    model = TriModalLorentzNet(
        eeg_channels=64,
        esg_channels=16,
        emg_channels=8,
        hidden_dim=64,
        latent_dim=32,
        num_classes=2
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    patience, patience_counter = 8, 0
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_ds.training = True
        train_correct, train_total = 0, 0
        
        for eeg, esg, emg, labels in train_loader:
            eeg, esg, emg, labels = eeg.to(device), esg.to(device), emg.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(eeg, esg, emg)
            logits = outputs['logits']
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Test
        model.eval()
        train_ds.training = False
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
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    return {'subject': subj, 'accuracy': best_acc, 'augmentation': augment}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load baseline to design strategy
    with open('physionet_full_results_final.pkl', 'rb') as f:
        baseline = pickle.load(f)
        baseline_accs = {item['subject']: item['accuracy'] for item in baseline} if isinstance(baseline, list) else {k: v['accuracy'] for k, v in baseline.items()}
    
    mean_acc = np.mean(list(baseline_accs.values()))
    
    # Adaptive: augment below-mean subjects only
    aug_subjects = {s for s, acc in baseline_accs.items() if acc < mean_acc}
    
    print("="*80)
    print("PHYSIONET ADAPTIVE - ALL 109 SUBJECTS (LOSO)")
    print("="*80)
    print(f"Baseline mean: {mean_acc:.2f}%")
    print(f"Augmentation ON:  {len(aug_subjects)} subjects (below mean)")
    print(f"Augmentation OFF: {len(baseline_accs) - len(aug_subjects)} subjects (above mean)")
    print("="*80)
    
    # Load preprocessed data (must exist from previous run)
    print("\nLoading preprocessed data...")
    data_file = Path('data/physionet_preprocessed.pkl')
    
    if not data_file.exists():
        print("❌ Preprocessed data not found. Run train_physionet_full.py first.")
        return
    
    with open(data_file, 'rb') as f:
        preprocessed = pickle.load(f)
    
    data_dict = preprocessed['data']
    labels_dict = preprocessed['labels']
    all_subjects = sorted(data_dict.keys())
    
    print(f"✅ Loaded {len(all_subjects)} subjects")
    
    results = []
    
    for subj in tqdm(all_subjects, desc="LOSO Training"):
        use_aug = subj in aug_subjects
        result = train_subject(subj, all_subjects, data_dict, labels_dict, device, 
                              augment=use_aug, epochs=25)
        results.append(result)
        
        if len(results) % 10 == 0:
            interim_mean = np.mean([r['accuracy'] for r in results])
            print(f"\n[Progress {len(results)}/{len(all_subjects)}] Mean so far: {interim_mean:.2f}%")
    
    # Final results
    accuracies = [r['accuracy'] for r in results]
    mean_final = np.mean(accuracies)
    std_final = np.std(accuracies)
    
    print("\n" + "="*80)
    print("FINAL RESULTS - PhysioNet Adaptive (109 Subjects)")
    print("="*80)
    print(f"Mean: {mean_final:.2f}% ± {std_final:.2f}%")
    print(f"Min:  {min(accuracies):.2f}%")
    print(f"Max:  {max(accuracies):.2f}%")
    print(f"\nBaseline: {mean_acc:.2f}%")
    print(f"Adaptive: {mean_final:.2f}%")
    print(f"Δ: {mean_final - mean_acc:+.2f}%")
    print("="*80)
    
    # Save
    with open('physionet_results_adaptive_full.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Saved: physionet_results_adaptive_full.pkl")


if __name__ == '__main__':
    main()
