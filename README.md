# LEGEND: Lorentzian EEG-ESG-EMG Graph Neural Network for Neural Bypass

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code and results for the CIBM submission **"LEGEND: A Tri-Modal Lorentzian Hyperbolic Graph Neural Network for Spinal Cord Injury Neural Bypass"** (manuscript ID CIBM-D-26-01579).

---

## Repository Layout

```
git-legend/
├── src/lorentz_tcnet/        # Core model source code
│   ├── model.py              # TriModalLorentzNet (main model)
│   ├── model_hybrid.py       # Hybrid variant used in LOSO experiments
│   ├── gnn_head.py           # Lorentzian GNN head
│   ├── graph.py              # Graph construction (cortico-spinal-muscular)
│   ├── train.py              # Training loop
│   ├── data.py               # Data loading & preprocessing
│   ├── metrics.py            # Evaluation metrics
│   └── config.py             # Hyperparameter defaults
├── scripts/                  # All experiment entry points
│   ├── train_hybrid_loso.py          # Main Steele LOSO experiment (Table 1)
│   ├── train_loso.py                 # Original LOSO baseline
│   ├── train_steele_baselines.py     # EEGNet / ShallowConvNet / BrainTopoGCN / EEG-GLT-Net / SAMGCN
│   ├── train_ablation_loso.py        # Ablations (EEG-only, no-EMG, no-ESG)
│   ├── train_bciciv2a.py             # BCI-IV-2a LOSO (clean, no leakage)
│   ├── train_physionet_full.py       # PhysioNet LOSO
│   ├── generate_publication_figures.py
│   └── ...
├── configs/
│   ├── default.yaml          # Steele dataset hyperparameters
│   └── physionet_config.yaml # PhysioNet hyperparameters
├── results/
│   ├── hybrid_loso_v7/       # LEGEND LOSO — all 10 folds (fold_result.json per fold)
│   ├── baselines_EEGNet/     # EEGNet LOSO results
│   ├── baselines_ShallowConvNet/
│   ├── baselines_BrainTopoGCN/
│   ├── baselines_EEG_GLT-Net/
│   ├── baselines_SAMGCN/
│   ├── euclidean_gnn_loso/   # Euclidean ablation (hyperbolic vs flat geometry)
│   ├── ablation_eeg_only/    # EEG-only ablation
│   ├── ablation_no_emg/      # No EMG ablation
│   ├── ablation_no_esg/      # No ESG ablation
│   ├── pathways_annotated.csv
│   └── cpu_latency.json
├── logs/                     # Raw training logs
├── *.pkl                     # Serialised per-fold result dictionaries
├── requirements.txt
└── pyproject.toml
```

---

## Environment Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

**Key dependencies:** PyTorch ≥ 2.1, torch-geometric, MNE, scikit-learn, numpy, scipy, matplotlib, pyyaml.

---

## Reproducing Results

### Data

The Steele et al. (2023) dataset (10 SCI participants, tri-modal EEG/ESG/EMG) is available from the [original authors](https://doi.org/10.1016/j.brs.2023.01.004). Place the `.mat` files under `data/steele_dataset/`.

BCI-IV-2a `.gdf` files: [BNCI Horizon 2020](http://bnci-horizon-2020.eu/database/data-sets).  
PhysioNet EEG MMIDB: `mne.datasets.eegbci.load_data()` (auto-downloaded by MNE).

### Main Tri-Modal LOSO Results (Table 1)

```bash
python scripts/train_hybrid_loso.py --config configs/default.yaml
```

Results are written to `results/hybrid_loso_v7/fold_NIS*/fold_result.json`.

### Baselines (Table 1 comparison rows)

```bash
python scripts/train_steele_baselines.py --model EEGNet
python scripts/train_steele_baselines.py --model ShallowConvNet
python scripts/train_steele_baselines.py --model BrainTopoGCN
python scripts/train_steele_baselines.py --model EEG-GLT-Net
python scripts/train_steele_baselines.py --model SAMGCN
```

### Ablations (Table 2)

```bash
python scripts/train_ablation_loso.py --ablation eeg_only
python scripts/train_ablation_loso.py --ablation no_emg
python scripts/train_ablation_loso.py --ablation no_esg
```

### BCI-IV-2a LOSO (Table 3)

```bash
python scripts/train_bciciv2a.py --loso
```

### PhysioNet LOSO (Table 3)

```bash
python scripts/train_physionet_full.py
```

---

## Pre-computed Results

All numerical results reported in the manuscript are available in `results/*/fold_result.json` and the `*.pkl` files at the repository root. Reviewers can verify every number without re-running training:

```python
import pickle, json, glob, numpy as np

# LEGEND LOSO on Steele dataset
folds = [json.load(open(f)) for f in sorted(glob.glob("results/hybrid_loso_v7/fold_*/fold_result.json"))]
accs = [f["test_acc"] for f in folds]
print(f"LEGEND LOSO: {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}%")

# PhysioNet
with open("physionet_full_results_final.pkl","rb") as fh:
    r = pickle.load(fh)
print(r)
```

---

## Citation

If you use this code or results, please cite:

```
[Manuscript under review — CIBM-D-26-01579]
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
