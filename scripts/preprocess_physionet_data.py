#!/usr/bin/env python3
"""Quick preprocessing of PhysioNet for adaptive training"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mne
from tqdm import tqdm
import pickle

def load_physionet_subject(subj_id, data_root='data/physionet/files'):
    """Load one PhysioNet subject from local directory"""
    subj_dir = Path(data_root) / f'S{subj_id:03d}'
    
    if not subj_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subj_dir}")
    
    # Motor imagery runs: 3,4 (hands), 7,8 (hands vs feet), 11,12 (hands)
    run_files = []
    for run in [3, 4, 7, 8, 11, 12]:
        run_file = subj_dir / f'S{subj_id:03d}R{run:02d}.edf'
        if run_file.exists():
            run_files.append(str(run_file))
    
    if len(run_files) == 0:
        raise FileNotFoundError(f"No valid run files for subject {subj_id}")
    
    raws = [mne.io.read_raw_edf(run, preload=True, verbose=False) for run in run_files]
    
    for raw in raws:
        raw.filter(1., 40., fir_design='firwin', verbose=False)
        raw.set_eeg_reference('average', projection=True, verbose=False)
    
    raw = mne.concatenate_raws(raws, verbose=False)
    
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    event_map = {v: k for k, v in event_id.items()}
    
    left_ids = [v for k, v in event_id.items() if 'T1' in k]
    right_ids = [v for k, v in event_id.items() if 'T2' in k]
    
    mask = np.isin(events[:, 2], left_ids + right_ids)
    events = events[mask]
    
    epochs = mne.Epochs(raw, events, tmin=0, tmax=4, baseline=None, 
                       preload=True, verbose=False, picks='eeg')
    
    X = epochs.get_data()
    
    # CRITICAL: Use epochs.events (post-drop), not original events
    kept_events = epochs.events
    y = np.array([0 if kept_events[i, 2] in left_ids else 1 
                  for i in range(len(kept_events))])
    
    # Verify match
    assert X.shape[0] == len(y), f"Data/label mismatch: {X.shape[0]} vs {len(y)}"
    
    return X, y

def main():
    print("Preprocessing PhysioNet data from local files...")
    
    # Check available subjects
    data_root = Path('data/physionet/files')
    if not data_root.exists():
        print(f"❌ Data directory not found: {data_root}")
        return
    
    available_subjects = sorted([int(d.name[1:]) for d in data_root.iterdir() 
                                if d.is_dir() and d.name.startswith('S')])
    
    print(f"Found {len(available_subjects)} subjects")
    
    data_dict = {}
    labels_dict = {}
    
    for subj_id in tqdm(available_subjects[:109], desc="Loading"):  # Use first 109
        try:
            X, y = load_physionet_subject(subj_id)
            subj_name = f'S{subj_id:03d}'
            data_dict[subj_name] = X
            labels_dict[subj_name] = y
        except Exception as e:
            print(f"⚠️  Subject {subj_id} failed: {e}")
            continue
    
    print(f"\n✅ Loaded {len(data_dict)} subjects")
    
    # Save
    Path('data').mkdir(exist_ok=True)
    with open('data/physionet_preprocessed.pkl', 'wb') as f:
        pickle.dump({'data': data_dict, 'labels': labels_dict}, f)
    
    print("💾 Saved: data/physionet_preprocessed.pkl")

if __name__ == '__main__':
    main()
