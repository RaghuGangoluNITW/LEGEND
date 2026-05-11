"""
Re-extract class-discriminative pathways from saved v3 checkpoints.

Uses the fixed hard one-hot prototype conditioning (true_label kwarg) so
each class's GNN attention is conditioned on its true prototype row —
not a soft average that collapses all classes to the same output.

Overwrites pathways_class*.csv in each fold directory and prints:
  - Mean |attn_class_i - attn_class_j| per fold (should be > 0)
  - Top-5 EEG-node Jaccard between every class pair (should be < 0.7)
  - Summary table at the end

Usage
-----
    python scripts/reextract_pathways.py [options]

    # Default: re-extract all folds from results/hybrid_loso_v3
    python scripts/reextract_pathways.py

    # Single fold for quick validation
    python scripts/reextract_pathways.py --subjects NIS008

    # Different results dir (e.g. after a new run)
    python scripts/reextract_pathways.py --results_dir results/hybrid_loso_v4
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from itertools import combinations

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lorentz_tcnet.model_hybrid import HyperLorentzNetHGCN
from src.lorentz_tcnet.pathway import PathwayExtractor

# ---------------------------------------------------------------------------
SUBJECTS = [f"NIS{i:03d}" for i in range(1, 11)]
N_EEG, N_ESG, N_EMG = 28, 15, 8


def parse_args():
    p = argparse.ArgumentParser(description="Re-extract class pathways from saved checkpoints")
    p.add_argument("--results_dir", default="results/hybrid_loso_v3",
                   help="Folder that contains fold_NIS*/best_model.pt")
    p.add_argument("--data_dir", default="data/steele",
                   help="Folder with NIS*.npz files")
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Subset of subjects to process (default: all 10)")
    # Model hyperparams — must match the saved checkpoint
    p.add_argument("--hidden_dim",  default=64,  type=int)
    p.add_argument("--latent_dim",  default=48,  type=int)
    p.add_argument("--gnn_hidden",  default=48,  type=int)
    p.add_argument("--gnn_layers",  default=1,   type=int)
    p.add_argument("--gnn_heads",   default=2,   type=int)
    p.add_argument("--dropout",     default=0.5, type=float)
    p.add_argument("--t_stride",    default=4,   type=int)
    p.add_argument("--batch_size",  default=64,  type=int)
    p.add_argument("--device",      default="cuda", type=str)
    p.add_argument("--top_k",       default=10,  type=int,
                   help="Top pathways to save per class")
    p.add_argument("--jaccard_k",   default=5,   type=int,
                   help="Top-k EEG nodes used for Jaccard discriminability check")
    return p.parse_args()


# ---------------------------------------------------------------------------
def load_subject(data_dir, subj):
    d = np.load(os.path.join(data_dir, f"{subj}.npz"), allow_pickle=True)
    return d["eeg"], d["esg"], d["emg"], d["labels"]


def infer_num_classes(state_dict: dict) -> int:
    """Read output dimension of stage2.classifier's last Linear weight."""
    key = "stage2.classifier.2.weight"
    if key in state_dict:
        return state_dict[key].shape[0]
    # Fallback: search for any Linear bias whose size is <= 8 (plausible n_classes)
    for k, v in state_dict.items():
        if k.endswith(".bias") and v.ndim == 1 and 2 <= v.shape[0] <= 8:
            return int(v.shape[0])
    raise RuntimeError("Cannot infer num_classes from checkpoint state dict. "
                       "Pass --num_classes explicitly or check key names.")


def jaccard(set_a, set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def top_k_eeg_nodes(avg_attn: np.ndarray, edge_index: np.ndarray,
                    n_eeg: int, k: int) -> set:
    """Return the k EEG source nodes with highest summed attention."""
    src = edge_index[0]
    eeg_mask = src < n_eeg
    eeg_scores = {}
    for e, s in enumerate(src):
        if s < n_eeg:
            eeg_scores[int(s)] = eeg_scores.get(int(s), 0.0) + avg_attn[e]
    top = sorted(eeg_scores, key=lambda x: eeg_scores[x], reverse=True)[:k]
    return set(top)


# ---------------------------------------------------------------------------
def reextract_fold(subj: str, args, device: torch.device) -> dict:
    fold_dir = os.path.join(args.results_dir, f"fold_{subj}")
    ckpt_path = os.path.join(fold_dir, "best_model.pt")

    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] {ckpt_path} not found")
        return {}

    print(f"\n{'='*60}")
    print(f"  Re-extracting: {subj}  ({fold_dir})")
    print(f"{'='*60}")

    # ---- Load checkpoint ----
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state   = ckpt["model_state"]
    ei      = ckpt["edge_index"]   # (2, E)
    ew      = ckpt["edge_weight"]  # (E,)

    num_classes = infer_num_classes(state)
    print(f"  Inferred num_classes = {num_classes}  |  edges = {ei.shape[1]}")

    # ---- Reconstruct model ----
    model = HyperLorentzNetHGCN(
        eeg_channels=N_EEG,
        esg_channels=N_ESG,
        emg_channels=N_EMG,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_classes=num_classes,
        gnn_hidden=args.gnn_hidden,
        gnn_layers=args.gnn_layers,
        gnn_heads=args.gnn_heads,
        dropout=args.dropout,
        use_stage1_logits=True,
        t_stride=args.t_stride,
    )
    model.register_graph(ei.to(device), ew.to(device))
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"  Model loaded  ({sum(p.numel() for p in model.parameters()):,} params)")

    # ---- Load TEST subject data (no augmentation — clean inference) ----
    eeg_te, esg_te, emg_te, labels_te = load_subject(args.data_dir, subj)
    print(f"  Test data: {eeg_te.shape}  labels: {np.unique(labels_te, return_counts=True)}")

    # Also pull a slice of TRAINING data so every class has enough samples
    # (test subject may be missing some classes)
    train_subjs = [s for s in SUBJECTS if s != subj]
    eeg_tr_parts, esg_tr_parts, emg_tr_parts, lab_tr_parts = [], [], [], []
    for ts in train_subjs:
        e, s, m, l = load_subject(args.data_dir, ts)
        # Take up to 30 trials per training subject (fast, low memory)
        n = min(30, len(l))
        eeg_tr_parts.append(e[:n])
        esg_tr_parts.append(s[:n])
        emg_tr_parts.append(m[:n])
        lab_tr_parts.append(l[:n])

    eeg_all = np.concatenate([eeg_te] + eeg_tr_parts, axis=0)
    esg_all = np.concatenate([esg_te] + esg_tr_parts, axis=0)
    emg_all = np.concatenate([emg_te] + emg_tr_parts, axis=0)
    lab_all = np.concatenate([labels_te] + lab_tr_parts, axis=0)

    print(f"  Combined pool: {len(lab_all)} trials  "
          f"classes present: {sorted(np.unique(lab_all).tolist())}")

    # ---- Accumulate attention with HARD prototype conditioning ----
    extractor = PathwayExtractor(
        model=model,
        edge_index=ei,
        n_eeg=N_EEG, n_esg=N_ESG, n_emg=N_EMG,
    )

    bs = args.batch_size
    n_total = len(lab_all)
    eeg_t = torch.from_numpy(eeg_all.astype(np.float32))
    esg_t = torch.from_numpy(esg_all.astype(np.float32))
    emg_t = torch.from_numpy(emg_all.astype(np.float32))
    lab_t = torch.from_numpy(lab_all.astype(np.int64))

    print(f"  Accumulating attention (hard labels)...")
    with torch.no_grad():
        for start in range(0, n_total, bs):
            end = min(start + bs, n_total)
            extractor.accumulate(
                (eeg_t[start:end], esg_t[start:end], emg_t[start:end], lab_t[start:end]),
                device=str(device),
            )

    # ---- Compute discriminability metrics ----
    classes_present = sorted(extractor._attn_count.keys())
    print(f"  Classes with accumulated attention: {classes_present}")

    attn_per_class = {c: extractor.avg_attention(c) for c in classes_present}

    # Mean pairwise |attn_i - attn_j|
    pair_diffs = []
    for ci, cj in combinations(classes_present, 2):
        d = np.abs(attn_per_class[ci] - attn_per_class[cj]).mean()
        pair_diffs.append((ci, cj, d))
    mean_diff = np.mean([d for _, _, d in pair_diffs]) if pair_diffs else 0.0

    # Top-k EEG Jaccard
    ei_np = ei.numpy()
    jaccards = []
    for ci, cj in combinations(classes_present, 2):
        nodes_i = top_k_eeg_nodes(attn_per_class[ci], ei_np, N_EEG, args.jaccard_k)
        nodes_j = top_k_eeg_nodes(attn_per_class[cj], ei_np, N_EEG, args.jaccard_k)
        jaccards.append(jaccard(nodes_i, nodes_j))
    mean_jacc = np.mean(jaccards) if jaccards else 1.0

    print(f"\n  [DISCRIMINABILITY CHECK]")
    for ci, cj, d in pair_diffs:
        print(f"    |attn_{ci} - attn_{cj}| mean = {d:.5f}")
    print(f"  Mean |attn diff| = {mean_diff:.5f}  (was 0.000 before fix; >0.001 = working)")
    for (ci, cj), j in zip(combinations(classes_present, 2), jaccards):
        print(f"    Jaccard top-{args.jaccard_k} EEG (class {ci} vs {cj}) = {j:.3f}")
    print(f"  Mean Jaccard = {mean_jacc:.3f}  (was 0.958 before fix; <0.7 = discriminative)")

    # ---- Save updated pathway CSVs (overwrites old broken ones) ----
    all_pathways = {}
    for cls in classes_present:
        try:
            pw = extractor.top_pathways(class_idx=cls, top_k=args.top_k)
            extractor.print_pathways(pw, class_name=f"Class {cls}")
            csv_path = os.path.join(fold_dir, f"pathways_class{cls}.csv")
            extractor.export_csv(pw, csv_path)
            all_pathways[cls] = pw
        except ValueError as e:
            print(f"  [Warning] {e}")

    # Save updated attn_per_class and discriminability metrics
    metrics = {
        "subject": subj,
        "mean_attn_diff": float(mean_diff),
        "mean_jaccard": float(mean_jacc),
        "pair_diffs": [(int(ci), int(cj), float(d)) for ci, cj, d in pair_diffs],
        "pairwise_jaccards": [(int(ci), int(cj), float(j))
                               for (ci, cj), j in zip(combinations(classes_present, 2),
                                                       jaccards)],
    }
    with open(os.path.join(fold_dir, "pathway_discriminability.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Update attn_per_class in fold_result.json if it exists
    res_path = os.path.join(fold_dir, "fold_result.json")
    if os.path.exists(res_path):
        with open(res_path) as f:
            res = json.load(f)
        res["attn_per_class"] = {str(c): attn_per_class[c].tolist()
                                 for c in classes_present}
        res["pathway_discriminability"] = metrics
        with open(res_path, "w") as f:
            json.dump(res, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Results dir: {args.results_dir}")
    print(f"Data dir:    {args.data_dir}")

    subjects = args.subjects if args.subjects else SUBJECTS
    print(f"Subjects: {subjects}")

    summary = []
    for subj in subjects:
        m = reextract_fold(subj, args, device)
        if m:
            summary.append(m)

    # ---- Summary table ----
    print(f"\n{'='*70}")
    print(f"  PATHWAY DISCRIMINABILITY SUMMARY  (results_dir: {args.results_dir})")
    print(f"{'='*70}")
    print(f"  {'Subject':<10} {'Mean |attn diff|':>18} {'Mean Jaccard':>14}  Status")
    print(f"  {'-'*10} {'-'*18} {'-'*14}  {'-'*20}")
    for m in summary:
        diff  = m["mean_attn_diff"]
        jacc  = m["mean_jaccard"]
        ok    = "OK - discriminative" if diff > 5e-4 and jacc < 0.7 else "CHECK - may be poor"
        print(f"  {m['subject']:<10} {diff:>18.5f} {jacc:>14.3f}  {ok}")

    if summary:
        all_diffs  = [m["mean_attn_diff"] for m in summary]
        all_jaccs  = [m["mean_jaccard"]   for m in summary]
        print(f"\n  MEAN across folds:")
        print(f"    Mean |attn diff| = {np.mean(all_diffs):.5f}")
        print(f"    Mean Jaccard     = {np.mean(all_jaccs):.3f}")

    print(f"\nDone. Updated pathways_class*.csv and pathway_discriminability.json "
          f"written to each fold dir.")


if __name__ == "__main__":
    main()
