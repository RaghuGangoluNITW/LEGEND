from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SubjectSample:
    subject_id: str
    eeg: np.ndarray
    esg: np.ndarray
    emg: np.ndarray
    label: int


class TriModalDataset(Dataset):
    def __init__(self, samples: List[SubjectSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "eeg": torch.as_tensor(sample.eeg, dtype=torch.float32),
            "esg": torch.as_tensor(sample.esg, dtype=torch.float32),
            "emg": torch.as_tensor(sample.emg, dtype=torch.float32),
            "labels": torch.as_tensor(sample.label, dtype=torch.long),  # Changed to 'labels' for consistency
        }


def _load_subject_file(file_path: Path) -> List[SubjectSample]:
    raw = np.load(file_path, allow_pickle=True)
    eeg = raw["eeg"]
    esg = raw["esg"]
    emg = raw["emg"]
    labels = raw["labels"]

    if not (len(eeg) == len(esg) == len(emg) == len(labels)):
        raise ValueError(f"Length mismatch in {file_path}")

    subject_id = file_path.stem
    samples: List[SubjectSample] = []
    for index in range(len(labels)):
        samples.append(
            SubjectSample(
                subject_id=subject_id,
                eeg=eeg[index],
                esg=esg[index],
                emg=emg[index],
                label=int(labels[index]),
            )
        )
    return samples


def load_subject_samples(dataset_dir: Path) -> Dict[str, List[SubjectSample]]:
    dataset_dir = Path(dataset_dir)  # Convert string to Path if needed
    subject_files = sorted(dataset_dir.glob("*.npz"))
    if not subject_files:
        raise FileNotFoundError(
            f"No subject files found in {dataset_dir}. Expected files like NIS001.npz"
        )

    subject_samples: Dict[str, List[SubjectSample]] = {}
    for file_path in subject_files:
        samples = _load_subject_file(file_path)
        if not samples:
            continue
        subject_samples[samples[0].subject_id] = samples

    if not subject_samples:
        raise RuntimeError("No valid samples were loaded.")

    return subject_samples


def build_loso_splits(subject_samples: Dict[str, List[SubjectSample]]) -> List[Tuple[TriModalDataset, TriModalDataset, str]]:
    """Build Leave-One-Subject-Out cross-validation splits.
    
    Returns:
        List of (train_dataset, val_dataset, test_subject_id) tuples
    """
    splits: List[Tuple[TriModalDataset, TriModalDataset, str]] = []
    subjects = sorted(subject_samples.keys())
    for test_subject in subjects:
        train_samples: List[SubjectSample] = []
        for subject_id, samples in subject_samples.items():
            if subject_id != test_subject:
                train_samples.extend(samples)
        test_samples = subject_samples[test_subject]
        
        # Wrap in datasets
        train_dataset = TriModalDataset(train_samples)
        val_dataset = TriModalDataset(test_samples)
        
        splits.append((train_dataset, val_dataset, test_subject))
    return splits
