"""
Ablation LOSO training for HyperLorentzNet-HGCN on Steele dataset.

Adds --modalities flag to zero out any subset of {eeg, esg, emg} signals,
enabling rigorous single-modality and leave-one-modality-out ablations.

Examples
--------
# Stage-1 EEG only (no ESG, no EMG, no graph head)
python scripts/train_ablation_loso.py \
    --modalities eeg --gnn_layers 0 \
    --results_dir results/ablation_eeg_only

# Full tri-modal, no ESG
python scripts/train_ablation_loso.py \
    --modalities eeg,emg \
    --results_dir results/ablation_no_esg

# Full tri-modal, no EMG
python scripts/train_ablation_loso.py \
    --modalities eeg,esg \
    --results_dir results/ablation_no_emg
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lorentz_tcnet.graph import TriLayerGraphBuilder
from src.lorentz_tcnet.model_hybrid import HyperLorentzNetHGCN

# ---------------------------------------------------------------------------
SUBJECTS = [f"NIS{i:03d}" for i in range(1, 11)]
N_EEG, N_ESG, N_EMG = 28, 15, 8


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation LOSO training")
    parser.add_argument("--data_dir",    default="data/steele", type=str)
    parser.add_argument("--results_dir", default="results/ablation", type=str)
    parser.add_argument("--modalities",  default="eeg,esg,emg", type=str,
                        help="Comma-separated active modalities from {eeg, esg, emg}")
    parser.add_argument("--epochs",      default=100, type=int)
    parser.add_argument("--batch_size",  default=32, type=int)
    parser.add_argument("--lr",          default=5e-4, type=float)
    parser.add_argument("--hidden_dim",  default=64, type=int)
    parser.add_argument("--latent_dim",  default=48, type=int)
    parser.add_argument("--gnn_hidden",  default=48, type=int)
    parser.add_argument("--gnn_layers",  default=2, type=int,
                        help="Set 0 to disable stage-2 HGCN (Stage-1 only)")
    parser.add_argument("--gnn_heads",   default=2, type=int)
    parser.add_argument("--k_cross",     default=6, type=int)
    parser.add_argument("--dropout",     default=0.5, type=float)
    parser.add_argument("--patience",    default=20, type=int)
    parser.add_argument("--t_stride",    default=4, type=int)
    parser.add_argument("--device",      default="cuda", type=str)
    parser.add_argument("--seed",        default=42, type=int)
    parser.add_argument("--resume",      action="store_true",
                        help="Skip folds that already have a fold_result.json")
    return parser.parse_args()


# ---------------------------------------------------------------------------
def load_subject(data_dir: str, subj: str):
    path = os.path.join(data_dir, f"{subj}.npz")
    d = np.load(path, allow_pickle=True)
    return d["eeg"], d["esg"], d["emg"], d["labels"]


class AugDataset(torch.utils.data.Dataset):
    def __init__(self, eeg, esg, emg, labels,
                 augment: bool = True,
                 prob: float = 0.25, shift: int = 7,
                 scale_range=(0.85, 1.15), noise_std: float = 0.007):
        self.eeg, self.esg, self.emg, self.labels = eeg, esg, emg, labels
        self.augment = augment
        self.prob, self.shift = prob, shift
        self.scale_range, self.noise_std = scale_range, noise_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = self.eeg[idx].copy()
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
            eeg = eeg + np.random.randn(*eeg.shape).astype(np.float32) * self.noise_std
        return (torch.from_numpy(eeg).float(),
                torch.from_numpy(esg).float(),
                torch.from_numpy(emg).float(),
                torch.tensor(lab, dtype=torch.long))


def make_loader(eeg, esg, emg, labels, batch_size, shuffle=True, augment=False, num_workers=0):
    dataset = AugDataset(eeg, esg, emg, labels, augment=augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=(num_workers == 0),
                      persistent_workers=False)


def stratified_val_split(eeg, esg, emg, labels, val_frac=0.20, seed=42):
    """Split one subject's data into train/val (stratified by class)."""
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
    """Class-stratified trial sample from training subjects for PLV construction."""
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


def apply_modality_mask(eeg_b, esg_b, emg_b, active: set):
    """Return None for modalities not in `active` so encoders are skipped entirely."""
    esg_b = esg_b if "esg" in active else None
    emg_b = emg_b if "emg" in active else None
    # eeg is always required by the model
    return eeg_b, esg_b, emg_b


# ---------------------------------------------------------------------------
def train_fold(test_subj, subjects, data_dir, results_dir, args, device, active_mods):
    print(f"\n{'='*60}")
    print(f"FOLD: hold-out = {test_subj}  |  modalities = {sorted(active_mods)}")
    print(f"{'='*60}")

    train_subjects = [s for s in subjects if s != test_subj]
    train_parts = [load_subject(data_dir, s) for s in train_subjects]
    labels_tr = np.concatenate([p[3] for p in train_parts], axis=0)
    eeg_te, esg_te, emg_te, labels_te = load_subject(data_dir, test_subj)

    num_classes = len(np.unique(labels_tr))
    n_train = sum(len(p[3]) for p in train_parts)
    print(f"  Train: {n_train} | Test: {eeg_te.shape[0]} | Classes: {num_classes}")

    # ---- PLV graph (class-stratified sample — R2.d fix) ----
    t0 = time.time()
    builder = TriLayerGraphBuilder(N_EEG, N_ESG, N_EMG)
    eeg_g, esg_g, emg_g = stratified_sample_plv(train_parts, n_total=256, seed=args.seed)
    # When a modality is zeroed, also zero the graph inputs so PLV=0 for
    # cross-layer edges involving that modality.
    if "eeg" not in active_mods:
        eeg_g = np.zeros_like(eeg_g)
    if "esg" not in active_mods:
        esg_g = np.zeros_like(esg_g)
    if "emg" not in active_mods:
        emg_g = np.zeros_like(emg_g)
    edge_index, edge_weight = builder.build(
        eeg_g, esg_g, emg_g,
        k_cross=args.k_cross, signed=True)
    del eeg_g, esg_g, emg_g
    print(f"  Graph: {edge_index.shape[1]} edges  ({time.time()-t0:.1f}s)")

    # ---- Model ----
    model = HyperLorentzNetHGCN(
        eeg_channels=N_EEG, esg_channels=N_ESG, emg_channels=N_EMG,
        hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
        num_classes=num_classes,
        gnn_hidden=args.gnn_hidden, gnn_layers=args.gnn_layers,
        gnn_heads=args.gnn_heads, dropout=args.dropout,
        use_stage1_logits=True, t_stride=args.t_stride,
    )
    model.register_graph(edge_index, edge_weight)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")

    # ---- Optimiser ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)
    class_counts = np.bincount(labels_tr, minlength=num_classes).astype(np.float32)
    class_counts = np.where(class_counts == 0, 1e6, class_counts)
    class_weights = torch.tensor(1.0 / (class_counts + 1e-6)).to(device)
    class_weights = class_weights / class_weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ---- Loaders: 80/20 trial-level stratified split (R2.b fix) ----
    nw = 0
    tr_splits, val_splits = [], []
    for eeg_s, esg_s, emg_s, lab_s in train_parts:
        split = stratified_val_split(eeg_s, esg_s, emg_s, lab_s,
                                     val_frac=0.20, seed=args.seed)
        tr_splits.append(split[:4])
        val_splits.append(split[4:])

    train_loader = DataLoader(
        ConcatDataset([AugDataset(t[0], t[1], t[2], t[3], augment=True)
                       for t in tr_splits]),
        batch_size=args.batch_size, shuffle=True, num_workers=nw,
        pin_memory=True, persistent_workers=False)

    val_eeg = np.concatenate([v[0] for v in val_splits])
    val_esg = np.concatenate([v[1] for v in val_splits])
    val_emg = np.concatenate([v[2] for v in val_splits])
    val_lab = np.concatenate([v[3] for v in val_splits])
    val_loader = make_loader(val_eeg, val_esg, val_emg, val_lab,
                             args.batch_size, shuffle=False, num_workers=nw)
    del val_eeg, val_esg, val_emg, val_lab

    # Test loader — only used ONCE after training
    test_loader = make_loader(eeg_te, esg_te, emg_te, labels_te,
                              args.batch_size, shuffle=False, augment=False,
                              num_workers=nw)

    # ---- AMP scaler ----
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ---- Training loop ----
    best_val_acc, best_state, patience_ctr = 0.0, None, 0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            eeg_b, esg_b, emg_b, lab_b = [t.to(device, non_blocking=True) for t in batch]
            eeg_b, esg_b, emg_b = apply_modality_mask(eeg_b, esg_b, emg_b, active_mods)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(eeg_b, esg_b, emg_b)
                loss = criterion(out["logits"], lab_b)
                if "stage1_logits" in out:
                    loss = loss + 0.2 * criterion(out["stage1_logits"], lab_b)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            for batch in val_loader:
                eeg_b, esg_b, emg_b, lab_b = [t.to(device, non_blocking=True) for t in batch]
                eeg_b, esg_b, emg_b = apply_modality_mask(eeg_b, esg_b, emg_b, active_mods)
                out = model(eeg_b, esg_b, emg_b)
                preds = out["logits"].argmax(-1)
                correct += (preds == lab_b).sum().item()
                total += lab_b.shape[0]
        val_acc = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or val_acc > best_val_acc:
            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  val={val_acc:.2f}%"
                  + ("  <-- best" if val_acc > best_val_acc else ""))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ---- Reload best-val model & evaluate on test subject ONCE ----
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for batch in test_loader:
            eeg_b, esg_b, emg_b, lab_b = [t.to(device, non_blocking=True) for t in batch]
            eeg_b, esg_b, emg_b = apply_modality_mask(eeg_b, esg_b, emg_b, active_mods)
            out = model(eeg_b, esg_b, emg_b)
            preds = out["logits"].argmax(-1)
            correct += (preds == lab_b).sum().item()
            total += lab_b.shape[0]
    test_acc = 100.0 * correct / total
    print(f"  Test acc (best-val model): {test_acc:.2f}%  |  best val: {best_val_acc:.2f}%")

    fold_dir = os.path.join(results_dir, f"fold_{test_subj}")
    os.makedirs(fold_dir, exist_ok=True)
    torch.save({"model_state": best_state,
                "edge_index": edge_index.cpu(),
                "edge_weight": edge_weight.cpu()},
               os.path.join(fold_dir, "best_model.pt"))
    result = {"subject": test_subj, "best_acc": test_acc,
              "best_val_acc": best_val_acc,
              "history": history, "n_params": total_params}
    with open(os.path.join(fold_dir, "fold_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  OK {test_subj}  test_acc = {test_acc:.2f}%")
    return result


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    active_mods = {m.strip().lower() for m in args.modalities.split(",")}
    valid = {"eeg", "esg", "emg"}
    if not active_mods.issubset(valid):
        raise ValueError(f"--modalities must be subset of {valid}, got: {active_mods}")
    print(f"Active modalities: {sorted(active_mods)}")
    print(f"GNN layers (stage-2): {args.gnn_layers}  "
          f"({'disabled - Stage-1 only' if args.gnn_layers == 0 else 'enabled'})")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU found.")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        print(f"Device: {device}  ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Device: {device}")

    os.makedirs(args.results_dir, exist_ok=True)

    fold_results = []
    for subj in SUBJECTS:
        fold_result_path = os.path.join(args.results_dir, f"fold_{subj}", "fold_result.json")
        if args.resume and os.path.exists(fold_result_path):
            with open(fold_result_path) as f:
                cached = json.load(f)
            print(f"  SKIP {subj}  (already done: {cached['best_acc']:.2f}%)")
            fold_results.append(cached)
            continue
        result = train_fold(subj, SUBJECTS, args.data_dir, args.results_dir,
                            args, device, active_mods)
        fold_results.append(result)

    accs = [r["best_acc"] for r in fold_results]
    mean_acc, std_acc = np.mean(accs), np.std(accs)
    print(f"\n{'='*60}")
    print(f"  ABLATION SUMMARY  (modalities={sorted(active_mods)}, "
          f"gnn_layers={args.gnn_layers})")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  {r['subject']}:  {r['best_acc']:.2f}%")
    print(f"  {'-'*40}")
    print(f"  Mean: {mean_acc:.2f}% +/- {std_acc:.2f}%")

    summary = {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "modalities": sorted(active_mods),
        "gnn_layers": args.gnn_layers,
        "per_subject": {r["subject"]: r["best_acc"] for r in fold_results},
        "args": vars(args),
    }
    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to: {args.results_dir}/")


if __name__ == "__main__":
    main()
