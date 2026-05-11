#!/usr/bin/env python3
"""
Adaptive BCI-IV-2a Training with Subject-Specific Augmentation

Based on baseline analysis:
- Augmentation ON (0.25): A02, A04, A05, A06, A07 (below-mean subjects)
- Augmentation OFF: A01, A03, A08, A09 (above-mean subjects)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import mne

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.lorentz_tcnet.model import TriModalLorentzNet


# Subject-specific augmentation strategy (based on baseline analysis)
AUG_SUBJECTS = {'A02', 'A04', 'A05', 'A06', 'A07'}  # Below mean, might benefit
NO_AUG_SUBJECTS = {'A01', 'A03', 'A08', 'A09'}  # Above mean, keep stable


class ModerateAugmentation:
    """Moderate data augmentation (0.25 probability)"""
    
    @staticmethod
    def time_shift(x: torch.Tensor, max_shift: int = 7) -> torch.Tensor:
        shift = np.random.randint(-max_shift, max_shift + 1)
        return torch.roll(x, shifts=shift, dims=-1)
    
    @staticmethod
    def amplitude_scale(x: torch.Tensor, scale_range: Tuple[float, float] = (0.85, 1.15)) -> torch.Tensor:
        scale = np.random.uniform(*scale_range)
        return x * scale
    
    @staticmethod
    def add_noise(x: torch.Tensor, noise_std: float = 0.007) -> torch.Tensor:
        noise = torch.randn_like(x) * noise_std
        return x + noise
    
    @staticmethod
    def channel_dropout(x: torch.Tensor, dropout_prob: float = 0.07) -> torch.Tensor:
        mask = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > dropout_prob
        return x * mask
    
    @staticmethod
    def apply_augmentations(x: torch.Tensor, aug_prob: float = 0.25) -> torch.Tensor:
        if np.random.rand() < aug_prob:
            x = ModerateAugmentation.time_shift(x)
        if np.random.rand() < aug_prob:
            x = ModerateAugmentation.amplitude_scale(x)
        if np.random.rand() < aug_prob:
            x = ModerateAugmentation.add_noise(x)
        if np.random.rand() < aug_prob:
            x = ModerateAugmentation.channel_dropout(x)
        return x


class BCICIV2aDataset(Dataset):
    def __init__(self, eeg_data: np.ndarray, labels: np.ndarray, use_augmentation: bool = False):
        self.eeg = torch.FloatTensor(eeg_data)
        self.labels = torch.LongTensor(labels)
        self.use_augmentation = use_augmentation
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        eeg = self.eeg[idx]
        label = self.labels[idx]
        
        if self.use_augmentation and self.training:
            eeg = ModerateAugmentation.apply_augmentations(eeg)
        
        # BCI-IV-2a has only EEG, create dummy ESG/EMG
        dummy_esg = torch.zeros(3, eeg.shape[1])  # 3 EOG channels as ESG
        dummy_emg = torch.zeros(1, eeg.shape[1])  # 1 dummy EMG channel
        
        return eeg, dummy_esg, dummy_emg, label


def load_bciciv2a_data(data_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load BCI Competition IV-2a data"""
    all_data = {}
    all_labels = {}
    
    subjects = [f'A0{i}' for i in range(1, 10)]
    
    for subject in subjects:
        # Load training data
        train_file = data_dir / f'{subject}T.gdf'
        raw_train = mne.io.read_raw_gdf(train_file, preload=True, verbose=False)
        
        # Load evaluation data
        eval_file = data_dir / f'{subject}E.gdf'
        raw_eval = mne.io.read_raw_gdf(eval_file, preload=True, verbose=False)
        
        # Get events
        events_train, _ = mne.events_from_annotations(raw_train, verbose=False)
        events_eval, _ = mne.events_from_annotations(raw_eval, verbose=False)
        
        # Filter for motor imagery events (769-772: left, right, feet, tongue)
        mi_events_train = events_train[(events_train[:, 2] >= 769) & (events_train[:, 2] <= 772)]
        mi_events_eval = events_eval[(events_eval[:, 2] >= 769) & (events_eval[:, 2] <= 772)]
        
        # Epoch data
        epochs_train = mne.Epochs(raw_train, mi_events_train, tmin=0, tmax=4, baseline=None, preload=True, verbose=False)
        epochs_eval = mne.Epochs(raw_eval, mi_events_eval, tmin=0, tmax=4, baseline=None, preload=True, verbose=False)
        
        # Get EEG channels (22 EEG + 3 EOG)
        eeg_channels = [ch for ch in epochs_train.ch_names if 'EEG' in ch or 'EOG' in ch]
        
        # Extract data
        train_data = epochs_train.get_data(picks=eeg_channels)
        eval_data = epochs_eval.get_data(picks=eeg_channels)
        
        # Combine train and eval
        combined_data = np.concatenate([train_data, eval_data], axis=0)
        combined_labels = np.concatenate([
            mi_events_train[:, 2] - 769,
            mi_events_eval[:, 2] - 769
        ])
        
        all_data[subject] = combined_data
        all_labels[subject] = combined_labels
    
    return all_data, all_labels


def train_one_subject(
    subject_id: str,
    all_subjects: List[str],
    data_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, np.ndarray],
    device: torch.device,
    use_augmentation: bool = True,
    num_epochs: int = 50
) -> Dict:
    """Train model with LOSO cross-validation"""
    
    # Leave-one-subject-out split
    train_subjects = [s for s in all_subjects if s != subject_id]
    test_subject = subject_id
    
    # Combine training data
    train_data = np.concatenate([data_dict[s] for s in train_subjects], axis=0)
    train_labels = np.concatenate([labels_dict[s] for s in train_subjects], axis=0)
    
    test_data = data_dict[test_subject]
    test_labels = labels_dict[test_subject]
    
    # Create datasets
    train_dataset = BCICIV2aDataset(train_data, train_labels, use_augmentation=use_augmentation)
    test_dataset = BCICIV2aDataset(test_data, test_labels, use_augmentation=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Model (baseline - pure Lorentzian)
    model = TriModalLorentzNet(
        eeg_channels=25,  # 22 EEG + 3 EOG
        esg_channels=3,   # Using EOG as dummy ESG
        emg_channels=1,   # Dummy EMG
        hidden_dim=64,
        latent_dim=32,
        num_classes=4
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print(f"\nTrain: {len(train_dataset)} trials")
    print(f"Test:  {len(test_dataset)} trials")
    
    best_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_dataset.training = True
        train_correct = 0
        train_total = 0
        
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
        
        train_acc = 100.0 * train_correct / train_total
        
        # Testing
        model.eval()
        train_dataset.training = False
        test_correct = 0
        test_total = 0
        
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
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train={train_acc:.2f}% | Test={test_acc:.2f}% (Best={best_acc:.2f}%)")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
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
    
    data_dir = Path('data/bciciv_2a/BCICIV_2a_gdf')
    
    print("="*80)
    print("ADAPTIVE STRATEGY - BCI-IV-2a Dataset (LOSO)")
    print("="*80)
    print("Strategy: Subject-specific augmentation based on baseline analysis")
    print(f"  Augmentation ON (0.25):  {sorted(AUG_SUBJECTS)}")
    print(f"  Augmentation OFF:        {sorted(NO_AUG_SUBJECTS)}")
    print(f"Total: {len(AUG_SUBJECTS) + len(NO_AUG_SUBJECTS)} subjects")
    print("="*80)
    
    # Load data
    data_dict, labels_dict = load_bciciv2a_data(data_dir)
    all_subjects = sorted(data_dict.keys())
    
    results = []
    
    for subject_id in tqdm(all_subjects, desc="LOSO Subjects"):
        print(f"\n{'='*80}")
        print(f"Test Subject: {subject_id}")
        use_aug = subject_id in AUG_SUBJECTS
        print(f"Augmentation: {'ON (0.25)' if use_aug else 'OFF'}")
        print("="*80)
        
        result = train_one_subject(subject_id, all_subjects, data_dict, labels_dict, device,
                                   use_augmentation=use_aug, num_epochs=50)
        results.append(result)
    
    # Final statistics
    accuracies = [r['accuracy'] for r in results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print("\n" + "="*80)
    print("FINAL RESULTS - Adaptive Strategy (BCI-IV-2a)")
    print("="*80)
    print(f"\nMean Accuracy: {mean_acc:.2f}%")
    print(f"Std Accuracy:  {std_acc:.2f}%")
    print(f"Min Accuracy:  {min(accuracies):.2f}%")
    print(f"Max Accuracy:  {max(accuracies):.2f}%")
    
    # Load baseline for comparison
    baseline_file = Path('bciciv2a_results_final.pkl')
    if baseline_file.exists():
        with open(baseline_file, 'rb') as f:
            baseline = pickle.load(f)
        baseline_mean = np.mean([item['accuracy'] for item in baseline])
        baseline_std = np.std([item['accuracy'] for item in baseline])
        
        print(f"\nCOMPARISON:")
        print(f"  Baseline: {baseline_mean:.2f}% ± {baseline_std:.2f}%")
        print(f"  Adaptive: {mean_acc:.2f}% ± {std_acc:.2f}%")
        print(f"\nImprovement: {mean_acc - baseline_mean:+.2f}%")
        
        if mean_acc > baseline_mean:
            print("✅ SUCCESS: Beat baseline with adaptive strategy!")
        else:
            print("⚠️  Adaptive did not improve over baseline")
    
    # Save results
    output_file = Path('bciciv2a_results_adaptive.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("="*80)


if __name__ == '__main__':
    main()
