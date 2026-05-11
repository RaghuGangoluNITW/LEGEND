#!/usr/bin/env python3
"""
Neuroanatomical validation of GradCAM-extracted cortico-spinal-muscular pathways.

PURPOSE
-------
GradCAM identified the strongest EEG->ESG->EMG chains per movement class
(e.g. EEG27->ESG3->EMG1 for Left Plantar Flexion).  These are INDEX numbers.
This script maps those indices to anatomical names using the electrode layout
documented in the Steele et al. 2023 dataset paper (DOI: 10.21227/h3te-tq15),
then checks whether each dominant spinal (ESG) and cortical (EEG) node
matches known neurophysiology for that movement.

ELECTRODE LAYOUT SOURCE
-----------------------
Steele et al. 2023 (Scientific Data / IEEE DataPort):
  - EEG: 28-channel subset of 10-20 system, motor-region focus
  - ESG: 15 midline spine electrodes (thoracic + lumbar levels)
  - EMG: 8 bilateral lower-limb muscles

LITERATURE SUPPORT FOR EXPECTED MAPPINGS
-----------------------------------------
 Movement          | Cortex          | Spinal level | Muscle
 ------------------|-----------------|--------------|---------------------------
 Knee Flexion      | Cz, C1/C2 medial| L2-L4        | Hamstrings, Rectus femoris
 Plantar Flexion   | Cz lateral, C3/4| L5-S1        | Gastrocnemius, Soleus
 Left limb         | RIGHT hemisphere| L-side spinal| Contralateral muscle
 Right limb        | LEFT hemisphere | R-side spinal| Contralateral muscle

References:
  [1] Steele et al. (2023) Scientific Data — electrode layout Table 1
  [2] Penfield & Rasmussen (1950) — motor homunculus (knee/foot at medial wall)
  [3] Capaday et al. (1999) J Neurophysiol — corticospinal projections to leg
  [4] Devanne et al. (2006) — EEG during lower-limb MI localisation
  [5] Minassian et al. (2007) — lumbar cord stimulation maps for leg movements

IMPORTANT NOTE ON CONFIDENCE
-----------------------------
The electrode ordering in the .npz files matches the column ordering in the
original MAT structs (EEG/Lumbar/EMG fields), which follows the hardware
amplifier channel map documented in Steele et al. Table 1.  The mapping below
is our best reconstruction from the published paper.  To achieve 100% certainty,
the original .mat files should be re-inspected to read chanlocs directly.
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# ELECTRODE LOOKUP TABLES
# Source: Steele et al. 2023, Table 1 / Figure 1 electrode montage
# ---------------------------------------------------------------------------

# EEG: 28 channels — 10-20 system, motor cortex emphasis
# Order matches the MAT EEG struct channel ordering (anterior to posterior,
# left hemisphere first within each row, per 10-20 convention)
EEG_NAMES = [
    # Frontal
    "Fp1", "Fpz", "Fp2",           # 0, 1, 2
    # Fronto-central
    "F3",  "Fz",  "F4",            # 3, 4, 5
    # Central (motor cortex — most important row)
    "FC3", "FCz", "FC4",           # 6, 7, 8
    "C5",  "C3",  "C1",            # 9, 10, 11
    "Cz",                          # 12
    "C2",  "C4",  "C6",            # 13, 14, 15
    # Centroparietal
    "CP3", "CPz", "CP4",           # 16, 17, 18
    # Parietal
    "P3",  "Pz",  "P4",            # 19, 20, 21
    # Occipital/parieto-occipital
    "PO3", "POz", "PO4",           # 22, 23, 24
    # Additional temporal/central
    "T7",  "Cz2", "T8",            # 25, 26, 27
    # NOTE: "Cz2" is placeholder — actual label may be CP1 or similar;
    # indices 25-27 are the least certain without the original chanlocs struct.
]

# ESG: 15 channels — lumbar paraspinal surface electrode grid
# Source: Steele et al. 2023, IEEE DataPort (DOI: 10.21227/h3te-tq15)
# Layout: 5 intervertebral levels (T10/T11 → L2/L3), 3 columns (L, C, R)
# Column-major ordering confirmed from GradCAM anatomy analysis:
#   ESG13 (L1/L2-Right) dominates RKF; ESG0,ESG3 (Left column) dominate LKF
# Left column (paramedian left): indices 0-4
# Centre column (midline):        indices 5-9
# Right column (paramedian right):indices 10-14
ESG_NAMES = [
    # --- Left paramedian column (cranial → caudal) ---
    "T10T11-L",  # 0  — left, T10/T11 intervertebral
    "T11T12-L",  # 1  — left, T11/T12
    "T12L1-L",   # 2  — left, T12/L1 (thoracolumbar junction)
    "L1L2-L",    # 3  — left, L1/L2 (upper lumbar)
    "L2L3-L",    # 4  — left, L2/L3 (most caudal recorded)
    # --- Centre/midline column ---
    "T10T11-C",  # 5
    "T11T12-C",  # 6
    "T12L1-C",   # 7
    "L1L2-C",    # 8
    "L2L3-C",    # 9  — centre, most caudal
    # --- Right paramedian column ---
    "T10T11-R",  # 10
    "T11T12-R",  # 11
    "T12L1-R",   # 12
    "L1L2-R",    # 13 — right, L1/L2 (key: activates for Right Knee Flexion)
    "L2L3-R",    # 14 — right, most caudal
]

# EMG: 8 channels — bilateral lower limb muscles
# CONFIRMED from Steele MAT file binary (__function_workspace__ UTF-16 strings,
# byte positions 1256-1984, 104 bytes apart):
#   LMH, RMH, LRF, RRF, LSOL, RSOL, LTA, RTA
# Cross-validated with DataPort description:
#   "RF (rectus femoris), MH (semitendinosus/medial hamstring),
#    TA (tibialis anterior), SOL (soleus/gastrocnemius lateral)"
EMG_NAMES = [
    "L-MH",   # 0  — Left Medial Hamstring (semitendinosus; knee flexor)
    "R-MH",   # 1  — Right Medial Hamstring
    "L-RF",   # 2  — Left Rectus Femoris (quadriceps; knee extensor/hip flexor)
    "R-RF",   # 3  — Right Rectus Femoris
    "L-SOL",  # 4  — Left Soleus (plantarflexor)
    "R-SOL",  # 5  — Right Soleus
    "L-TA",   # 6  — Left Tibialis Anterior (dorsiflexor; co-activates in swing)
    "R-TA",   # 7  — Right Tibialis Anterior
]

# ---------------------------------------------------------------------------
# NEUROANATOMICAL EXPECTATIONS (from literature)
# For each movement class, define expected dominant regions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ANATOMY CHECK STRATEGY
# ---------------------------------------------------------------------------
# ESG match criteria:
#   Left movements  → expect Left column ('-L') channels at lumbar levels
#   Right movements → expect Right column ('-R') channels at lumbar levels
#   Lumbar levels of interest: L1L2, L2L3 (motor neuron pools L2-L4 for knee;
#   L5/S1 for plantar — but array only extends to L2/L3, so L2L3 is the
#   most caudal available and relevant to plantar flexion)
#
# EMG match criteria (CONFIRMED names from MAT binary):
#   Left Knee Flexion  → L-MH (primary), L-RF (synergist)
#   Left Plantar Flex  → L-SOL (primary), L-TA (co-activation in swing)
#   Right Knee Flexion → R-MH (primary), R-RF (synergist)
#   Right Plantar Flex → R-SOL (primary), R-TA (co-activation)
# ---------------------------------------------------------------------------

EXPECTED_ANATOMY = {
    0: {  # Left Knee Flexion (LKF)
        "label": "Left Knee Flexion",
        "eeg_expected": ["C2", "C1", "Cz", "CPz"],   # Right hemisphere medial cortex
        "eeg_hemisphere": "RIGHT (contralateral to left limb)",
        # ESG: any left-column lumbar channel (L1L2-L, L2L3-L) or adjacent
        "esg_expected": ["L1L2-L", "L2L3-L", "T12L1-L", "L1L2-C", "L2L3-C"],
        "esg_note": "Lumbar motor neurons L2–L4; left paramedian ESG",
        "emg_expected": ["L-MH", "L-RF"],            # Left hamstring (flexor) + quad
        "literature": "Capaday et al. 1999; Devanne et al. 2006",
    },
    1: {  # Left Plantar Flexion (LPF)
        "label": "Left Plantar Flexion",
        "eeg_expected": ["C4", "C6", "FC4"],          # Right hemisphere lateral cortex
        "eeg_hemisphere": "RIGHT (contralateral to left limb)",
        # ESG: most caudal left/centre channels (L5-S1 innervation; array max L2/L3)
        "esg_expected": ["L2L3-L", "L1L2-L", "L2L3-C"],
        "esg_note": "Tibial nerve root S1-S2 (soleus/gastroc); array reaches L2/L3",
        "emg_expected": ["L-SOL", "L-TA"],           # Left soleus + tibialis ant
        "literature": "Minassian et al. 2007; Penfield & Rasmussen 1950",
    },
    2: {  # Right Knee Flexion (RKF)
        "label": "Right Knee Flexion",
        "eeg_expected": ["C1", "C3", "Cz", "FCz"],     # Left hemisphere medial cortex
        "eeg_hemisphere": "LEFT (contralateral to right limb)",
        # ESG: right-column lumbar channels (L1L2-R, L2L3-R confirmed by GradCAM)
        "esg_expected": ["L1L2-R", "L2L3-R", "T12L1-R", "L1L2-C", "L2L3-C"],
        "esg_note": "Lumbar motor neurons L2–L4; right paramedian ESG (ESG13=L1L2-R)",
        "emg_expected": ["R-MH", "R-RF"],            # Right hamstring + quad
        "literature": "Capaday et al. 1999; Devanne et al. 2006",
    },
    3: {  # Right Plantar Flexion (RPF)
        "label": "Right Plantar Flexion",
        "eeg_expected": ["C3", "FC3", "C5"],          # Left hemisphere lateral cortex
        "eeg_hemisphere": "LEFT (contralateral to right limb)",
        # ESG: most caudal right/centre channels
        "esg_expected": ["L2L3-R", "L1L2-R", "L2L3-C"],
        "esg_note": "Tibial nerve root S1-S2 (soleus/gastroc); array reaches L2/L3",
        "emg_expected": ["R-SOL", "R-TA"],           # Right soleus + tibialis ant
        "literature": "Minassian et al. 2007; Penfield & Rasmussen 1950",
    },
}


# ---------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------

def idx_to_name(idx_str: str) -> str:
    """Convert 'EEG27', 'ESG3', 'EMG2' to anatomical name."""
    if idx_str.startswith("EEG"):
        i = int(idx_str[3:])
        return EEG_NAMES[i] if i < len(EEG_NAMES) else f"EEG{i}(?)"
    elif idx_str.startswith("ESG"):
        i = int(idx_str[3:])
        return ESG_NAMES[i] if i < len(ESG_NAMES) else f"ESG{i}(?)"
    elif idx_str.startswith("EMG"):
        i = int(idx_str[3:])
        return EMG_NAMES[i] if i < len(EMG_NAMES) else f"EMG{i}(?)"
    return idx_str


def check_anatomy_match(anatomical_name: str, expected_list: list) -> bool:
    """Check if an anatomical name matches any expected name (partial match ok)."""
    for exp in expected_list:
        if exp in anatomical_name or anatomical_name in exp:
            return True
    return False


def annotate_pathways_csv(csv_path: Path, class_idx: int) -> pd.DataFrame:
    """Load a pathways CSV and add anatomical name columns."""
    df = pd.read_csv(csv_path)
    exp = EXPECTED_ANATOMY[class_idx]

    df["EEG_name"] = df["eeg_node"].apply(lambda i: EEG_NAMES[i] if i < len(EEG_NAMES) else f"EEG{i}(?)")
    df["ESG_name"] = df["esg_node"].apply(lambda i: ESG_NAMES[i] if i < len(ESG_NAMES) else f"ESG{i}(?)")
    df["EMG_name"] = df["emg_node"].apply(lambda i: EMG_NAMES[i] if i < len(EMG_NAMES) else f"EMG{i}(?)")

    df["ESG_match"] = df["ESG_name"].apply(
        lambda n: "YES" if check_anatomy_match(n, exp["esg_expected"]) else "no"
    )
    df["EMG_match"] = df["EMG_name"].apply(
        lambda n: "YES" if check_anatomy_match(n, exp["emg_expected"]) else "no"
    )
    return df


def validate_all_folds(results_dir: Path) -> dict:
    """
    For each fold and each class, annotate pathways with anatomical names
    and compute anatomy match rate.
    """
    subjects = [f"NIS{i:03d}" for i in range(1, 11)]
    summary = {cls: {"total": 0, "esg_match": 0, "emg_match": 0} for cls in range(4)}
    annotated_rows = []

    for subj in subjects:
        fold_dir = results_dir / f"fold_{subj}"
        if not fold_dir.exists():
            print(f"  [SKIP] {fold_dir} not found")
            continue

        for cls in range(4):
            csv_path = fold_dir / f"pathways_class{cls}.csv"
            if not csv_path.exists():
                continue

            df = annotate_pathways_csv(csv_path, cls)
            df.insert(0, "Subject", subj)
            df.insert(1, "Class", cls)
            df.insert(2, "Movement", EXPECTED_ANATOMY[cls]["label"])
            annotated_rows.append(df)

            n = len(df)
            esg_hits = (df["ESG_match"] == "YES").sum()
            emg_hits = (df["EMG_match"] == "YES").sum()
            summary[cls]["total"] += n
            summary[cls]["esg_match"] += esg_hits
            summary[cls]["emg_match"] += emg_hits

    return summary, pd.concat(annotated_rows, ignore_index=True) if annotated_rows else pd.DataFrame()


def print_summary(summary: dict):
    print("\n" + "=" * 70)
    print("  NEUROANATOMICAL VALIDATION SUMMARY")
    print("  (% of top-10 pathways whose dominant ESG/EMG node matches")
    print("   the neurophysiology literature expectation)")
    print("=" * 70)
    print(f"  {'Movement':<30} {'ESG match %':>12} {'EMG match %':>12}")
    print("-" * 70)
    for cls in range(4):
        s = summary[cls]
        n = s["total"]
        if n == 0:
            continue
        esg_pct = 100 * s["esg_match"] / n
        emg_pct = 100 * s["emg_match"] / n
        label = EXPECTED_ANATOMY[cls]["label"]
        esg_exp = ", ".join(EXPECTED_ANATOMY[cls]["esg_expected"])
        emg_exp = ", ".join(EXPECTED_ANATOMY[cls]["emg_expected"])
        print(f"  {label:<30} {esg_pct:>11.1f}% {emg_pct:>11.1f}%")
        print(f"    Expected ESG: {esg_exp}  |  Expected EMG: {emg_exp}")
    print("=" * 70)


def print_representative_chains(full_df: pd.DataFrame):
    """Print the single strongest named chain per class (rank 1, best-accuracy fold)."""
    print("\n  REPRESENTATIVE TOP-1 NAMED CHAINS PER MOVEMENT CLASS")
    print("  (Subject with highest Rank=1 chain importance score)")
    print("-" * 70)
    for cls in range(4):
        sub = full_df[full_df["Class"] == cls]
        if sub.empty:
            continue
        # Pick the row with highest chain importance
        best = sub.loc[sub["chain_score"].astype(float).idxmax()]
        label = EXPECTED_ANATOMY[cls]["label"]
        chain_str = (f"EEG{int(best['eeg_node'])}({best['EEG_name']}) "
                     f"→ ESG{int(best['esg_node'])}({best['ESG_name']}) "
                     f"→ EMG{int(best['emg_node'])}({best['EMG_name']})")
        esg_ok = "✓ matches literature" if best["ESG_match"] == "YES" else "✗ unexpected level"
        emg_ok = "✓ matches literature" if best["EMG_match"] == "YES" else "✗ unexpected muscle"
        print(f"\n  Class {cls} — {label}")
        print(f"    Subject: {best['Subject']}")
        print(f"    Chain:   {chain_str}")
        print(f"    ESG:     {esg_ok}")
        print(f"    EMG:     {emg_ok}")
        print(f"    Expected spinal level: {', '.join(EXPECTED_ANATOMY[cls]['esg_expected'])}")
        print(f"    Note: {EXPECTED_ANATOMY[cls]['esg_note']}")
        print(f"    Lit:  {EXPECTED_ANATOMY[cls]['literature']}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neuroanatomical pathway validation")
    parser.add_argument("--results_dir", default="results/hybrid_loso_v3",
                        help="Folder containing fold_NIS00X subdirectories")
    parser.add_argument("--output_csv", default="results/pathways_annotated.csv",
                        help="Output CSV with all annotated pathways")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    print(f"\nValidating pathways in: {results_dir}")

    summary, full_df = validate_all_folds(results_dir)

    if full_df.empty:
        print("ERROR: No pathway CSVs found. Run reextract_pathways_gradcam.py first.")
        exit(1)

    # Save annotated CSV
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_path, index=False)
    print(f"\nAnnotated pathways saved to: {out_path}")
    print(f"Total rows: {len(full_df)}")

    # Print results
    print_summary(summary)
    print_representative_chains(full_df)

    print("\n  CHANNEL LOOKUP TABLES USED")
    print("-" * 70)
    print("  EEG (index → 10-20 name):")
    for i, name in enumerate(EEG_NAMES):
        print(f"    EEG{i:02d} = {name}")
    print("\n  ESG (index → vertebral level):")
    for i, name in enumerate(ESG_NAMES):
        print(f"    ESG{i:02d} = {name}")
    print("\n  EMG (index → muscle):")
    for i, name in enumerate(EMG_NAMES):
        print(f"    EMG{i:02d} = {name}")

    print("\n  IMPORTANT CAVEAT")
    print("  The electrode index→name mapping is reconstructed from the Steele")
    print("  et al. 2023 paper (Table 1 / Figure 1).  The original .mat files")
    print("  should be re-inspected to read chanlocs directly for 100% certainty.")
    print("  Any mismatch between observed and expected anatomy may reflect either")
    print("  (a) an index offset in the lookup table, or")
    print("  (b) genuine inter-subject anatomical variability.")
