# LEGEND: Lorentzian Electro-modal Graph Encoder for Neural Decoding for SCI Rehabilitation

**Manuscript:** "LEGEND: Lorentzian Electro-modal Graph Encoder for Neural Decoding for SCI Rehabilitation"  
**Journal:** Computers in Biology and Medicine

This repository contains the complete source code and all pre-computed results needed to verify the results reported in the manuscript **without re-running training**.

---

## Repository Structure

```
git-legend/
├── src/
│   └── lorentz_tcnet/              # Core model implementation
│       ├── model_hybrid.py         # TriModalLorentzNet (main model, Table 1)
│       ├── model.py                # Base model
│       ├── gnn_head.py             # Lorentzian hyperbolic GNN head
│       ├── graph.py                # Cortico-spinal-muscular graph builder
│       ├── train.py                # Training loop & early stopping
│       ├── data.py                 # Dataset loading & windowing
│       ├── metrics.py              # Accuracy, balanced accuracy
│       └── config.py               # Hyperparameter defaults
│
├── scripts/
│   ├── train_hybrid_loso.py        # → Table 1: LEGEND LOSO (Steele dataset)
│   ├── train_steele_baselines.py   # → Table 1: EEGNet / ShallowConvNet / BrainTopoGCN / EEG-GLT-Net / SAMGCN
│   ├── train_ablation_loso.py      # → Table 2: EEG-only / no-EMG / no-ESG ablations
│   ├── train_euclidean_gnn_loso.py # → Table 2: Euclidean GNN ablation (Lorentzian vs flat)
│   ├── train_bciciv2a.py           # → Table 3: BCI-IV-2a LOSO (cross-subject)
│   ├── train_physionet_full.py     # → Table 3: PhysioNet LOSO
│   ├── convert_steele_mat.py       # Data: convert .mat files to numpy arrays
│   ├── preprocess_physionet_data.py # Data: preprocess PhysioNet EEG
│   ├── reextract_pathways.py       # Figure 5: pathway attribution (Grad-CAM)
│   ├── validate_pathway_anatomy.py # Figure 5: anatomical plausibility check
│   ├── generate_publication_figures.py # All manuscript figures
│   └── measure_cpu_latency.py      # Table 4: inference latency
│
├── configs/
│   ├── default.yaml                # Steele dataset hyperparameters
│   └── physionet_config.yaml       # PhysioNet hyperparameters
│
├── results/
│   ├── hybrid_loso_v7/fold_NIS*/fold_result.json   # LEGEND per-fold results (Table 1)
│   ├── baselines_EEGNet/           # EEGNet LOSO results
│   ├── baselines_ShallowConvNet/   # ShallowConvNet LOSO results
│   ├── baselines_BrainTopoGCN/     # BrainTopoGCN LOSO results
│   ├── baselines_EEG_GLT-Net/      # EEG-GLT-Net LOSO results
│   ├── baselines_SAMGCN/           # SAMGCN LOSO results
│   ├── euclidean_gnn_loso/         # Euclidean GNN ablation (Table 2 last row)
│   ├── ablation_eeg_only/          # EEG-only ablation
│   ├── ablation_no_emg/            # No EMG ablation
│   ├── ablation_no_esg/            # No ESG ablation
│   ├── pathways_annotated.csv      # Figure 5 data
│   └── cpu_latency.json            # Table 4 latency numbers
│
├── physionet_full_results_final.pkl    # PhysioNet LOSO results (Table 3)
├── physionet_results_adaptive_full.pkl # PhysioNet adaptive results
├── bciciv2a_results_final.pkl          # BCI-IV-2a LOSO results (Table 3)
├── steele_results_phase4.pkl           # Steele training history
├── steele_results_enhanced.pkl         # Steele enhanced training history
├── logs/                           # Training logs (BCI-IV-2a, PhysioNet)
├── requirements.txt
└── pyproject.toml
```

---

## Verifying Results Without Re-Training

All manuscript numbers can be verified directly from the pre-computed JSON files:

```python
import json, glob, numpy as np, pickle

# ── Table 1: LEGEND LOSO on Steele dataset ──────────────────────────────────
folds = [json.load(open(f))
         for f in sorted(glob.glob("results/hybrid_loso_v7/fold_NIS*/fold_result.json"))]
accs = [f["test_acc"] for f in folds]
print(f"LEGEND LOSO (Steele): {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}%")
# Expected: 56.51 ± 12.27%

# ── Table 1: EEGNet baseline ─────────────────────────────────────────────────
baseline = json.load(open("results/baselines_EEGNet/loso_summary.json"))
print(f"EEGNet LOSO: {baseline['mean_acc']*100:.2f} ± {baseline['std_acc']*100:.2f}%")

# ── Table 2: Euclidean GNN ablation ──────────────────────────────────────────
euc = json.load(open("results/euclidean_gnn_loso/loso_summary.json"))
print(f"EuclideanGNN LOSO: {euc['mean_acc']*100:.2f}%  (vs LEGEND {np.mean(accs)*100:.2f}%)")

# ── Table 3: PhysioNet & BCI-IV-2a ───────────────────────────────────────────
with open("physionet_full_results_final.pkl", "rb") as fh:
    physio = pickle.load(fh)
print("PhysioNet:", physio)

with open("bciciv2a_results_final.pkl", "rb") as fh:
    bci = pickle.load(fh)
print("BCI-IV-2a:", bci)
```

---

## Reproducing Results from Scratch

### 1. Environment

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data

| Dataset | Source | Notes |
|---|---|---|
| Steele et al. 2023 (tri-modal EEG/ESG/EMG, 10 SCI subjects) | Contact original authors (DOI: 10.1016/j.brs.2023.01.004) | Place `.mat` files in `data/steele_dataset/` |
| BCI Competition IV Dataset 2a | [bnci-horizon-2020.eu](http://bnci-horizon-2020.eu/database/data-sets) | Place `.gdf` files in `data/bciciv_2a/` |
| PhysioNet EEG Motor Movement/Imagery (EEGMMI) | Auto-downloaded by MNE | Run `python scripts/preprocess_physionet_data.py` |

Convert Steele `.mat` to numpy:
```bash
python scripts/convert_steele_mat.py --input data/steele_dataset/ --output data/steele/
```

### 3. Re-run Each Experiment

```bash
# Table 1 — LEGEND LOSO (main result)
python scripts/train_hybrid_loso.py --config configs/default.yaml

# Table 1 — All 5 baselines
python scripts/train_steele_baselines.py --model EEGNet
python scripts/train_steele_baselines.py --model ShallowConvNet
python scripts/train_steele_baselines.py --model BrainTopoGCN
python scripts/train_steele_baselines.py --model EEG-GLT-Net
python scripts/train_steele_baselines.py --model SAMGCN

# Table 2 — Ablations
python scripts/train_ablation_loso.py --ablation eeg_only
python scripts/train_ablation_loso.py --ablation no_emg
python scripts/train_ablation_loso.py --ablation no_esg
python scripts/train_euclidean_gnn_loso.py

# Table 3 — Cross-dataset
python scripts/train_bciciv2a.py --loso          # BCI-IV-2a LOSO
python scripts/train_physionet_full.py            # PhysioNet LOSO

# Figure 5 — Pathway attributions
python scripts/reextract_pathways.py
python scripts/validate_pathway_anatomy.py

# All figures
python scripts/generate_publication_figures.py
```

### 4. Hardware

All experiments were run on a single consumer GPU (NVIDIA GeForce RTX 2050, 4 GB VRAM). Each LOSO fold trains in approximately 5–10 minutes. Full 10-fold LOSO takes 50–100 minutes per experiment.

---

## Key Results (from pre-computed files)

| Experiment | Result | Table |
|---|---|---|
| LEGEND tri-modal LOSO (Steele) | **56.51 ± 12.27%** | Table 1 |
| EEGNet LOSO | 35.1% | Table 1 |
| ShallowConvNet LOSO | 37.8% | Table 1 |
| BrainTopoGCN LOSO | 40.2% | Table 1 |
| EEG-GLT-Net LOSO | 42.1% | Table 1 |
| SAMGCN LOSO | 44.8% | Table 1 |
| EEG-only ablation | ~38% | Table 2 |
| Euclidean GNN LOSO | 48.94 ± 13.43% | Table 2 |
| BCI-IV-2a LOSO (first published) | **42.9 ± 7.0%** | Table 3 |
| PhysioNet LOSO | **80.5 ± 10.5%** | Table 3 |

---

## Citation

> [Manuscript under review  citation will be added upon acceptance]

---

## License

MIT License. See [LICENSE](LICENSE).
