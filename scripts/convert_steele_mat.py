#!/usr/bin/env python3
"""
Convert Steele dataset MAT files to NPZ format for Lorentz TCN training.

This script:
1. Loads raw .mat files with nested trial structure
2. Extracts EEG (28 ch), ESG (15 ch), EMG (8 ch) data
3. Segments continuous data into fixed-length windows
4. Saves as .npz with keys: 'eeg', 'esg', 'emg', 'labels'

Label mapping:
  0 = LKF (Left Knee Flexion)
  1 = LPF (Left Plantar Flexion)
  2 = RKF (Right Knee Flexion)
  3 = RPF (Right Plantar Flexion)
"""

import argparse
import numpy as np
import scipy.io
from pathlib import Path
from typing import Dict, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SteeleMatConverter:
    """Convert Steele MAT files to segmented NPZ format."""
    
    # Label mapping
    LABEL_MAP = {
        'LKF': 0,  # Left Knee Flexion
        'LPF': 1,  # Left Plantar Flexion
        'RKF': 2,  # Right Knee Flexion
        'RPF': 3,  # Right Plantar Flexion
    }
    
    def __init__(self, window_size: int = 1000, stride: int = 500, 
                 min_window_size: int = 800):
        """
        Initialize converter.
        
        Args:
            window_size: Number of samples per window (default: 1000 = 1 sec @ 1000 Hz)
            stride: Stride between windows (default: 500 = 50% overlap)
            min_window_size: Minimum samples for a valid window (discard shorter)
        """
        self.window_size = window_size
        self.stride = stride
        self.min_window_size = min_window_size
        
    def segment_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Segment continuous signal into fixed-length windows.
        
        Args:
            signal: Array of shape (channels, timepoints)
            
        Returns:
            Array of shape (n_windows, channels, window_size)
        """
        n_channels, n_samples = signal.shape
        
        # Calculate number of windows
        n_windows = (n_samples - self.window_size) // self.stride + 1
        
        if n_windows <= 0:
            # Handle short signals
            if n_samples >= self.min_window_size:
                # Pad to window_size
                pad_width = self.window_size - n_samples
                padded = np.pad(signal, ((0, 0), (0, pad_width)), mode='edge')
                return padded[np.newaxis, :, :]  # Shape: (1, channels, window_size)
            else:
                # Signal too short, return empty
                return np.empty((0, n_channels, self.window_size))
        
        # Create sliding windows
        windows = []
        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            if end_idx <= n_samples:
                windows.append(signal[:, start_idx:end_idx])
        
        if not windows:
            return np.empty((0, n_channels, self.window_size))
            
        return np.stack(windows, axis=0)  # Shape: (n_windows, channels, window_size)
    
    def load_subject_mat(self, mat_path: Path) -> Dict[str, np.ndarray]:
        """
        Load subject MAT file and extract trial data.
        
        Args:
            mat_path: Path to .mat file (e.g., NIS001.mat)
            
        Returns:
            Dictionary with keys 'eeg', 'esg', 'emg', 'labels' (all segmented)
        """
        logger.info(f"Loading {mat_path.name}...")
        
        # Load MAT file
        data = scipy.io.loadmat(str(mat_path))
        
        # Extract subject data (key is NIS1, NIS2, etc.)
        subject_key = list(k for k in data.keys() if k.startswith('NIS'))[0]
        subject_data = data[subject_key][0, 0]
        
        # Extract sampling rate (for logging)
        srate = subject_data['srate'][0, 0]
        logger.info(f"  Sampling rate: {srate} Hz")
        
        # Extract modality structures
        lumbar_struct = subject_data['Lumbar'][0, 0]  # ESG data
        eeg_struct = subject_data['EEG'][0, 0]
        emg_struct = subject_data['EMG'][0, 0]
        
        # Collect segmented data from all trials
        all_eeg = []
        all_esg = []
        all_emg = []
        all_labels = []
        
        for trial_name, label_idx in self.LABEL_MAP.items():
            # Extract continuous data for this trial
            esg_continuous = lumbar_struct[trial_name]  # Shape: (15, timepoints)
            eeg_continuous = eeg_struct[trial_name]     # Shape: (28, timepoints)
            emg_continuous = emg_struct[trial_name]     # Shape: (8, timepoints)
            
            # Verify shapes match
            assert esg_continuous.shape[1] == eeg_continuous.shape[1] == emg_continuous.shape[1], \
                f"Timepoint mismatch for {trial_name}"
            
            n_samples = eeg_continuous.shape[1]
            logger.info(f"  {trial_name}: {n_samples} samples ({n_samples/srate:.1f} sec)")
            
            # Segment into windows
            eeg_windows = self.segment_signal(eeg_continuous)
            esg_windows = self.segment_signal(esg_continuous)
            emg_windows = self.segment_signal(emg_continuous)
            
            n_windows = eeg_windows.shape[0]
            
            if n_windows > 0:
                all_eeg.append(eeg_windows)
                all_esg.append(esg_windows)
                all_emg.append(emg_windows)
                all_labels.append(np.full(n_windows, label_idx, dtype=np.int64))
                logger.info(f"    → {n_windows} windows extracted")
            else:
                logger.warning(f"    → No windows extracted (too short)")
        
        # Concatenate all trials
        if not all_eeg:
            raise ValueError(f"No valid windows extracted from {mat_path.name}")
        
        eeg_data = np.concatenate(all_eeg, axis=0)  # (N, 28, window_size)
        esg_data = np.concatenate(all_esg, axis=0)  # (N, 15, window_size)
        emg_data = np.concatenate(all_emg, axis=0)  # (N, 8, window_size)
        labels = np.concatenate(all_labels, axis=0)  # (N,)
        
        logger.info(f"  Total: {len(labels)} windows across {len(self.LABEL_MAP)} classes")
        logger.info(f"  Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        return {
            'eeg': eeg_data.astype(np.float32),
            'esg': esg_data.astype(np.float32),
            'emg': emg_data.astype(np.float32),
            'labels': labels
        }
    
    def convert_subject(self, mat_path: Path, output_dir: Path) -> Path:
        """
        Convert single subject MAT file to NPZ.
        
        Args:
            mat_path: Path to input .mat file
            output_dir: Directory to save .npz file
            
        Returns:
            Path to saved .npz file
        """
        # Extract subject data
        subject_data = self.load_subject_mat(mat_path)
        
        # Save as NPZ
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{mat_path.stem}.npz"
        
        np.savez_compressed(
            output_path,
            eeg=subject_data['eeg'],
            esg=subject_data['esg'],
            emg=subject_data['emg'],
            labels=subject_data['labels']
        )
        
        logger.info(f"  Saved: {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        return output_path
    
    def convert_all_subjects(self, input_dir: Path, output_dir: Path) -> None:
        """
        Convert all MAT files in directory to NPZ format.
        
        Args:
            input_dir: Directory containing .mat files
            output_dir: Directory to save .npz files
        """
        mat_files = sorted(input_dir.glob("*.mat"))
        
        if not mat_files:
            raise ValueError(f"No .mat files found in {input_dir}")
        
        logger.info("="*80)
        logger.info(f"Converting {len(mat_files)} subjects from Steele dataset")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Window size: {self.window_size} samples")
        logger.info(f"Stride: {self.stride} samples (overlap: {(1 - self.stride/self.window_size)*100:.0f}%)")
        logger.info("="*80)
        
        for mat_file in mat_files:
            logger.info(f"\n[{mat_files.index(mat_file)+1}/{len(mat_files)}] Processing {mat_file.name}...")
            try:
                self.convert_subject(mat_file, output_dir)
            except Exception as e:
                logger.error(f"  Failed to convert {mat_file.name}: {e}")
                raise
        
        logger.info("\n" + "="*80)
        logger.info("✅ Conversion complete!")
        logger.info("="*80)
        
        # Verify output
        npz_files = sorted(output_dir.glob("*.npz"))
        logger.info(f"Output: {len(npz_files)} NPZ files")
        for npz_file in npz_files:
            data = np.load(npz_file)
            logger.info(f"  {npz_file.name}: {len(data['labels'])} samples, "
                       f"classes {np.unique(data['labels']).tolist()}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Steele MAT files to NPZ format for Lorentz TCN training"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/steele_dataset"),
        help="Directory containing .mat files (default: data/steele_dataset)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/steele"),
        help="Directory to save .npz files (default: data/steele)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="Window size in samples (default: 1000 = 1 sec @ 1000 Hz)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=500,
        help="Stride between windows (default: 500 = 50%% overlap)"
    )
    parser.add_argument(
        "--min-window-size",
        type=int,
        default=800,
        help="Minimum window size to keep (default: 800 samples)"
    )
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = SteeleMatConverter(
        window_size=args.window_size,
        stride=args.stride,
        min_window_size=args.min_window_size
    )
    
    # Convert all subjects
    converter.convert_all_subjects(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
