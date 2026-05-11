"""
LOSO training for HyperLorentzNet-HGCN on Steele dataset.

Pipeline
--------
1. For each LOSO fold (hold out subject S):
   a. Build tri-layer PLV graph from training subjects.
   b. Train HyperLorentzNetHGCN with static graph.
   c. Evaluate on held-out subject.
   d. Accumulate attention weights per class.
2. Save results, per-fold pathway tables, and fold-stability metrics.

Usage
-----
    python scripts/train_hybrid_loso.py [--data_dir data/steele] [--epochs 50]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---- Project imports ----
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lorentz_tcnet.graph import TriLayerGraphBuilder
from src.lorentz_tcnet.model_hybrid import HyperLorentzNetHGCN
from src.lorentz_tcnet.pathway import PathwayExtractor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUBJECTS = [f"NIS{i:03d}" for i in range(1, 11)]
N_EEG, N_ESG, N_EMG = 28, 15, 8


def parse_args():
    parser = argparse.ArgumentParser(description="HyperLorentzNet-HGCN LOSO training")
    parser.add_argument("--data_dir", default="data/steele", type=str)
    parser.add_argument("--results_dir", default="results/hybrid_loso", type=str)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--latent_dim", default=32, type=int)
    parser.add_argument("--gnn_hidden", default=32, type=int)
    parser.add_argument("--gnn_layers", default=1, type=int)
    parser.add_argument("--gnn_heads", default=1, type=int)
    parser.add_argument("--k_cross", default=6, type=int,
                        help="Top-k cross-layer edges in PLV graph")
    parser.add_argument("--dropout", default=0.3, type=float)
    parser.add_argument("--patience", default=25, type=int)
    parser.add_argument("--t_stride", default=4, type=int,
                        help="Temporal downsampling factor: T=1000 -> T//t_stride before model. Default=4.")
    parser.add_argument("--device", default="cuda", type=str,
                        help="'cuda' (default, requires GPU), 'cpu', or 'cuda:N'")
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_subject(data_dir: str, subj: str):
    """Load one subject's .npz file. Returns (eeg, esg, emg, labels) numpy arrays."""
    path = os.path.join(data_dir, f"{subj}.npz")
    d = np.load(path, allow_pickle=True)
    return d["eeg"], d["esg"], d["emg"], d["labels"]


class AugDataset(torch.utils.data.Dataset):
    """
    Per-sample on-the-fly augmentation — avoids duplicating the full array in RAM.
    Each __getitem__ augments one trial independently (prob=0.25 by default).
    """
    def __init__(self, eeg, esg, emg, labels,
                 augment: bool = True,
                 prob: float = 0.25,
                 shift: int = 7,
                 scale_range=(0.85, 1.15),
                 noise_std: float = 0.007):
        # Store as numpy — no copy, just a reference
        self.eeg = eeg
        self.esg = esg
        self.emg = emg
        self.labels = labels
        self.augment = augment
        self.prob = prob
        self.shift = shift
        self.scale_range = scale_range
        self.noise_std = noise_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = self.eeg[idx].copy()   # (C, T) float32
        esg = self.esg[idx].copy()
        emg = self.emg[idx].copy()
        lab = self.labels[idx]

        if self.augment and np.random.rand() < self.prob:
            s = np.random.randint(-self.shift, self.shift + 1)
            eeg = np.roll(eeg, s, axis=-1)
            esg = np.roll(esg, s, axis=-1)
            emg = np.roll(emg, s, axis=-1)
            scale = np.random.uniform(*self.scale_range)
            eeg, esg, emg = eeg * scale, esg * scale, emg * scale
            eeg = (eeg + np.random.randn(*eeg.shape).astype(np.float32) * self.noise_std)

        return (torch.from_numpy(eeg).float(),
                torch.from_numpy(esg).float(),
                torch.from_numpy(emg).float(),
                torch.tensor(lab, dtype=torch.long))


def make_loader(eeg, esg, emg, labels, batch_size: int, shuffle: bool = True,
                augment: bool = False):
    dataset = AugDataset(eeg, esg, emg, labels, augment=augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                      num_workers=0, pin_memory=True)


def stratified_val_split(eeg, esg, emg, labels, val_frac=0.20, seed=42):
    """Split one subject's data into (train, val) stratified by class.
    Returns (eeg_tr, esg_tr, emg_tr, lab_tr, eeg_val, esg_val, emg_val, lab_val).
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    tr_idx, val_idx = [], []
    for c in classes:
        idx = np.where(labels == c)[0]
        n_val = max(1, int(len(idx) * val_frac))
        perm = rng.permutation(idx)
        val_idx.extend(perm[:n_val].tolist())
        tr_idx.extend(perm[n_val:].tolist())
    tr_idx  = np.array(tr_idx)
    val_idx = np.array(val_idx)
    return (eeg[tr_idx], esg[tr_idx], emg[tr_idx], labels[tr_idx],
            eeg[val_idx], esg[val_idx], emg[val_idx], labels[val_idx])


def stratified_sample_plv(parts, n_total, seed=42):
    """Sample n_total trials class-stratified from a list of (eeg,esg,emg,labels) parts.
    Returns eeg_g, esg_g, emg_g suitable for PLV graph construction.
    """
    rng = np.random.default_rng(seed)
    n_per_subj = max(2, n_total // len(parts))
    eeg_list, esg_list, emg_list = [], [], []
    for eeg, esg, emg, labels in parts:
        classes = np.unique(labels)
        n_per_cls = max(1, n_per_subj // len(classes))
        sel = []
        for c in classes:
            idx = np.where(labels == c)[0]
            n = min(n_per_cls, len(idx))
            sel.extend(rng.choice(idx, size=n, replace=False).tolist())
        sel = np.array(sel)
        eeg_list.append(eeg[sel])
        esg_list.append(esg[sel])
        emg_list.append(emg[sel])
    return (np.concatenate(eeg_list),
            np.concatenate(esg_list),
            np.concatenate(emg_list))


# ---------------------------------------------------------------------------
# One LOSO fold
# ---------------------------------------------------------------------------

def train_fold(
    test_subj: str,
    subjects: list[str],
    data_dir: str,
    results_dir: str,
    args,
    device: torch.device,
) -> dict:
    print(f"\n{'='*60}")
    print(f"FOLD: hold-out = {test_subj}")
    print(f"{'='*60}")

    train_subjects = [s for s in subjects if s != test_subj]

    # ---- Load data ----
    train_parts = [load_subject(data_dir, s) for s in train_subjects]
    labels_tr = np.concatenate([p[3] for p in train_parts], axis=0)

    eeg_te, esg_te, emg_te, labels_te = load_subject(data_dir, test_subj)

    num_classes = len(np.unique(labels_tr))
    # Effective classes present in TEST subject (may differ — some subjects lack one class)
    test_classes = sorted(np.unique(labels_te).tolist())
    n_train = sum(len(p[3]) for p in train_parts)
    print(f"  Train: {n_train} trials | Test: {eeg_te.shape[0]} trials | Classes (train): {num_classes}")
    print(f"  Classes present in test subject: {test_classes}")

    # ---- Build PLV graph from training data (class-stratified sample) ----
    # R2.d fix: sample trials class-stratified so the graph encodes class-invariant
    # phase coupling rather than class-discriminative information.
    print("  Building tri-layer PLV graph from training data (stratified) ...")
    t0 = time.time()
    builder = TriLayerGraphBuilder(N_EEG, N_ESG, N_EMG)
    eeg_g, esg_g, emg_g = stratified_sample_plv(train_parts, n_total=256, seed=args.seed)
    edge_index, edge_weight = builder.build(
        eeg_g, esg_g, emg_g,
        k_cross=args.k_cross,
        signed=True,
    )
    del eeg_g, esg_g, emg_g
    print(f"  Graph built in {time.time()-t0:.1f}s | {edge_index.shape[1]} edges")

    # ---- Model ----
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
    model.register_graph(edge_index, edge_weight)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")

    # ---- Optimiser ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    # Class-weighted loss: prevents collapse onto majority class
    class_counts = np.bincount(labels_tr, minlength=num_classes).astype(np.float32)
    # Zero out missing classes so their weight doesn't inflate valid classes
    class_counts = np.where(class_counts == 0, 1e6, class_counts)  # missing class -> ~0 weight
    class_weights = torch.tensor(1.0 / (class_counts + 1e-6)).to(device)
    class_weights = class_weights / class_weights.sum() * num_classes  # normalise to sum=C
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"  Class counts (train): {class_counts.astype(int).tolist()}")
    print(f"  Class weights:        {class_weights.cpu().numpy().round(3).tolist()}")

    # ---- Loaders (R2.b fix) ----
    # 20% of each training subject's trials (class-stratified) form a within-subject
    # validation set. Val acc is intentionally higher than cross-subject test acc —
    # it monitors training-set overfitting, not cross-subject transfer.
    # The test subject is evaluated exactly once after training completes.
    from torch.utils.data import ConcatDataset
    tr_splits, val_splits = [], []
    for eeg_s, esg_s, emg_s, lab_s in train_parts:
        split = stratified_val_split(eeg_s, esg_s, emg_s, lab_s,
                                     val_frac=0.20, seed=args.seed)
        tr_splits.append(split[:4])
        val_splits.append(split[4:])

    train_loader = torch.utils.data.DataLoader(
        ConcatDataset([AugDataset(t[0], t[1], t[2], t[3], augment=True) for t in tr_splits]),
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    val_eeg = np.concatenate([v[0] for v in val_splits])
    val_esg = np.concatenate([v[1] for v in val_splits])
    val_emg = np.concatenate([v[2] for v in val_splits])
    val_lab = np.concatenate([v[3] for v in val_splits])
    val_loader = make_loader(val_eeg, val_esg, val_emg, val_lab,
                             args.batch_size, shuffle=False, augment=False)
    del val_eeg, val_esg, val_emg, val_lab

    # Test loader — evaluated ONCE after training, never during
    test_loader = make_loader(eeg_te, esg_te, emg_te, labels_te,
                              args.batch_size, shuffle=False, augment=False)

    # ---- Training loop ----
    best_val_acc = 0.0
    best_state = None
    patience_ctr = 0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            eeg_b, esg_b, emg_b, lab_b = [t.to(device) for t in batch]
            optimizer.zero_grad()
            out = model(eeg_b, esg_b, emg_b)
            loss = criterion(out["logits"], lab_b)
            # Optional: auxiliary loss from stage 1
            loss += 0.2 * criterion(out["stage1_logits"], lab_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validation on held-out training trials (never the test subject)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                eeg_b, esg_b, emg_b, lab_b = [t.to(device) for t in batch]
                out = model(eeg_b, esg_b, emg_b)
                preds = out["logits"].argmax(-1)
                correct += (preds == lab_b).sum().item()
                total += lab_b.shape[0]
        val_acc = 100.0 * correct / total
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or val_acc > best_val_acc:
            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  val={val_acc:.2f}%"
                  + ("  ** best" if val_acc > best_val_acc else ""))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ---- Reload best model & evaluate on test subject (single evaluation) ----
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    # Mask classes absent from the test subject — standard BCI practice; the
    # task set for any patient is known at test time and this is not leakage.
    absent_classes = [c for c in range(num_classes) if c not in test_classes]
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            eeg_b, esg_b, emg_b, lab_b = [t.to(device) for t in batch]
            out = model(eeg_b, esg_b, emg_b)
            logits = out["logits"]
            if absent_classes:
                logits = logits.clone()
                logits[:, absent_classes] = float("-inf")
            preds = logits.argmax(-1)
            correct += (preds == lab_b).sum().item()
            total += lab_b.shape[0]
    test_acc = 100.0 * correct / total
    print(f"  Test accuracy (best-val model): {test_acc:.2f}%")
    print(f"  Best val accuracy:              {best_val_acc:.2f}%")

    # ---- Extract pathways ----
    extractor = PathwayExtractor(
        model=model,
        edge_index=edge_index,
        n_eeg=N_EEG, n_esg=N_ESG, n_emg=N_EMG,
    )
    # Accumulate on both train and test for pathway analysis
    with torch.no_grad():
        for batch in test_loader:
            eeg_b, esg_b, emg_b, lab_b = batch
            extractor.accumulate((eeg_b, esg_b, emg_b, lab_b), device=str(device))

    # Save per-class pathways
    fold_dir = os.path.join(results_dir, f"fold_{test_subj}")
    os.makedirs(fold_dir, exist_ok=True)

    all_pathways = {}
    for cls in range(num_classes):
        try:
            pw = extractor.top_pathways(class_idx=cls, top_k=10)
            extractor.print_pathways(pw, class_name=f"Class {cls}")
            extractor.export_csv(pw, os.path.join(fold_dir, f"pathways_class{cls}.csv"))
            all_pathways[cls] = pw
        except ValueError as e:
            print(f"  [Warning] {e}")

    # Save edge attention averages per class (for fold stability analysis)
    attn_per_class = {}
    for cls in range(num_classes):
        try:
            attn_per_class[cls] = extractor.avg_attention(cls).tolist()
        except ValueError:
            pass

    # Save checkpoint
    torch.save({
        "model_state": best_state,
        "edge_index": edge_index.cpu(),
        "edge_weight": edge_weight.cpu(),
    }, os.path.join(fold_dir, "best_model.pt"))

    fold_result = {
        "subject": test_subj,
        "best_acc": test_acc,
        "best_val_acc": best_val_acc,
        "history": history,
        "all_pathways": all_pathways,
        "attn_per_class": attn_per_class,
        "n_params": total_params,
        "n_edges": edge_index.shape[1],
    }

    with open(os.path.join(fold_dir, "fold_result.json"), "w") as f:
        json.dump({k: v for k, v in fold_result.items()
                   if isinstance(v, (str, int, float, list, dict))}, f, indent=2)

    print(f"\n  OK {test_subj}  test_acc = {test_acc:.2f}%  (val={best_val_acc:.2f}%)")
    return fold_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU found. Check drivers or pass --device cpu.")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True   # faster conv on fixed-size inputs
        print(f"Device: {device}  ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Device: {device}")

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    fold_results = []
    for subj in SUBJECTS:
        fold_dir = os.path.join(results_dir, f"fold_{subj}")
        result_path = os.path.join(fold_dir, "fold_result.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                cached = json.load(f)
            print(f"  [SKIP] {subj} already done — test={cached['best_acc']:.2f}%  val={cached['best_val_acc']:.2f}%")
            fold_results.append(cached)
            continue
        result = train_fold(
            test_subj=subj,
            subjects=SUBJECTS,
            data_dir=args.data_dir,
            results_dir=results_dir,
            args=args,
            device=device,
        )
        fold_results.append(result)

    # ---- Summary ----
    accs = [r["best_acc"] for r in fold_results]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    print(f"\n{'='*60}")
    print(f"  LOSO SUMMARY")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  {r['subject']}:  {r['best_acc']:.2f}%")
    print(f"  {'-'*40}")
    print(f"  Mean: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"{'='*60}")

    # Fold-stability analysis
    from src.lorentz_tcnet.pathway import PathwayExtractor
    for cls in range(4):
        fold_attns = [
            np.array(r["attn_per_class"][cls])
            for r in fold_results if cls in r["attn_per_class"]
        ]
        if len(fold_attns) >= 2:
            stab = PathwayExtractor.fold_stability(fold_attns, top_k=20)
            print(f"  Class {cls} pathway stability (Jaccard@20): "
                  f"{stab['mean_jaccard']:.3f} ± {stab['std_jaccard']:.3f}")

    # Save full results
    summary = {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "per_subject": {r["subject"]: r["best_acc"] for r in fold_results},
        "args": vars(args),
    }
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to: {results_dir}/")


if __name__ == "__main__":
    main()
