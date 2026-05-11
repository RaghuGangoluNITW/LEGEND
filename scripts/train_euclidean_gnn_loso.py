"""
Euclidean GNN ablation (R1.3) — LOSO on Steele EEG+ESG+EMG dataset.

This script is the direct Euclidean counterpart of train_hybrid_loso.py.
Everything is held constant EXCEPT the graph head:
  - HyperLorentzNetHGCN:  Lorentz projection + Lorentzian GAT + Fréchet mean pooling
  - EuclideanTriModalGNN: No projection + standard Euclidean GAT + mean pooling

This isolates the contribution of hyperbolic geometry to classification accuracy.

Usage:
    python scripts/train_euclidean_gnn_loso.py --results_dir results/euclidean_gnn_loso
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lorentz_tcnet.graph import TriLayerGraphBuilder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUBJECTS = [f"NIS{i:03d}" for i in range(1, 11)]
N_EEG, N_ESG, N_EMG = 28, 15, 8


def parse_args():
    p = argparse.ArgumentParser(description="Euclidean GNN ablation LOSO training")
    p.add_argument("--data_dir",    default="data/steele")
    p.add_argument("--results_dir", default="results/euclidean_gnn_loso")
    p.add_argument("--epochs",      default=50,   type=int)
    p.add_argument("--batch_size",  default=32,   type=int)
    p.add_argument("--lr",          default=5e-4, type=float)
    p.add_argument("--hidden_dim",  default=64,   type=int)
    p.add_argument("--latent_dim",  default=32,   type=int)
    p.add_argument("--gnn_hidden",  default=32,   type=int)
    p.add_argument("--gnn_layers",  default=1,    type=int)
    p.add_argument("--gnn_heads",   default=1,    type=int)
    p.add_argument("--k_cross",     default=6,    type=int)
    p.add_argument("--dropout",     default=0.3,  type=float)
    p.add_argument("--patience",    default=25,   type=int)
    p.add_argument("--t_stride",    default=4,    type=int)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--seed",        default=42,   type=int)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers (identical to train_hybrid_loso.py)
# ---------------------------------------------------------------------------

def load_subject(data_dir: str, subj: str):
    path = os.path.join(data_dir, f"{subj}.npz")
    d = np.load(path, allow_pickle=True)
    return d["eeg"], d["esg"], d["emg"], d["labels"]


class AugDataset(torch.utils.data.Dataset):
    def __init__(self, eeg, esg, emg, labels, augment=True,
                 prob=0.25, shift=7, scale_range=(0.85, 1.15), noise_std=0.007):
        self.eeg, self.esg, self.emg, self.labels = eeg, esg, emg, labels
        self.augment = augment
        self.prob = prob
        self.shift = shift
        self.scale_range = scale_range
        self.noise_std = noise_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = self.eeg[idx].copy()
        esg = self.esg[idx].copy()
        emg = self.emg[idx].copy()
        lab = self.labels[idx]
        if self.augment and np.random.rand() < self.prob:
            s = np.random.randint(-self.shift, self.shift + 1)
            eeg, esg, emg = np.roll(eeg, s, -1), np.roll(esg, s, -1), np.roll(emg, s, -1)
            scale = np.random.uniform(*self.scale_range)
            eeg, esg, emg = eeg * scale, esg * scale, emg * scale
            eeg = eeg + np.random.randn(*eeg.shape).astype(np.float32) * self.noise_std
        return (torch.from_numpy(eeg).float(), torch.from_numpy(esg).float(),
                torch.from_numpy(emg).float(), torch.tensor(lab, dtype=torch.long))


def make_loader(eeg, esg, emg, labels, batch_size, shuffle=True, augment=False):
    return DataLoader(AugDataset(eeg, esg, emg, labels, augment=augment),
                      batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


def stratified_val_split(eeg, esg, emg, labels, val_frac=0.20, seed=42):
    rng = np.random.default_rng(seed)
    tr_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        n_val = max(1, int(len(idx) * val_frac))
        perm = rng.permutation(idx)
        val_idx.extend(perm[:n_val].tolist())
        tr_idx.extend(perm[n_val:].tolist())
    ti, vi = np.array(tr_idx), np.array(val_idx)
    return (eeg[ti], esg[ti], emg[ti], labels[ti],
            eeg[vi], esg[vi], emg[vi], labels[vi])


def stratified_sample_plv(parts, n_total, seed=42):
    rng = np.random.default_rng(seed)
    n_per_subj = max(2, n_total // len(parts))
    eeg_l, esg_l, emg_l = [], [], []
    for eeg, esg, emg, labels in parts:
        classes = np.unique(labels)
        n_per_cls = max(1, n_per_subj // len(classes))
        sel = []
        for c in classes:
            idx = np.where(labels == c)[0]
            sel.extend(rng.choice(idx, size=min(n_per_cls, len(idx)), replace=False).tolist())
        sel = np.array(sel)
        eeg_l.append(eeg[sel]); esg_l.append(esg[sel]); emg_l.append(emg[sel])
    return np.concatenate(eeg_l), np.concatenate(esg_l), np.concatenate(emg_l)


# ---------------------------------------------------------------------------
# Euclidean GNN model
# ---------------------------------------------------------------------------

class TemporalBlock(nn.Module):
    """Identical to src.lorentz_tcnet.model.TemporalBlock."""
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1  = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1    = nn.BatchNorm1d(out_ch)
        self.drop1  = nn.Dropout(dropout)
        self.conv2  = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2    = nn.BatchNorm1d(out_ch)
        self.drop2  = nn.Dropout(dropout)
        self.resid  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        res = x if self.resid is None else self.resid(x)
        out = self.conv1(x)[:, :, :x.shape[-1]]
        out = self.drop1(F.relu(self.bn1(out)))
        out = self.conv2(out)[:, :, :x.shape[-1]]
        out = self.drop2(F.relu(self.bn2(out)))
        return F.relu(out + res)


class ModalityEncoder(nn.Module):
    """TCN encoder: (B, C, T) -> (B, latent_dim). Identical to the Lorentzian Stage 1."""
    def __init__(self, in_channels, hidden_dim, latent_dim, dropout=0.2):
        super().__init__()
        self.stem = nn.Conv1d(in_channels, hidden_dim, 3, padding=1)
        self.tcn1 = TemporalBlock(hidden_dim, hidden_dim, dilation=1, dropout=dropout)
        self.tcn2 = TemporalBlock(hidden_dim, hidden_dim, dilation=2, dropout=dropout)
        self.tcn3 = TemporalBlock(hidden_dim, hidden_dim, dilation=4, dropout=dropout)
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.stem(x))
        x = self.tcn3(self.tcn2(self.tcn1(x)))
        return self.proj(x.mean(-1))    # (B, latent_dim)


class EuclideanGATLayer(nn.Module):
    """
    Standard single-head (or multi-head) Graph Attention layer.
    Euclidean counterpart of LorentzGraphAttentionLayer.

    Input : x (B, N, in_dim)
    Output: x (B, N, out_dim * heads)
    """
    def __init__(self, in_dim, out_dim, heads=1, dropout=0.2):
        super().__init__()
        self.heads   = heads
        self.out_dim = out_dim
        self.W_val   = nn.Linear(in_dim, out_dim * heads, bias=False)
        # Attention: for each head, a learnable weight vector a ∈ R^{2*out_dim}
        self.a       = nn.Parameter(torch.empty(heads, 2 * out_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)
        self.bn      = nn.BatchNorm1d(out_dim * heads)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        """
        x:           (B, N, in_dim)
        edge_index:  (2, E)  — src (row) -> dst (col)
        edge_weight: (E,)    — signed PLV weights (scalar per edge)
        Returns:     (B, N, out_dim * heads)
        """
        B, N, _ = x.shape
        H, D = self.heads, self.out_dim
        src, dst = edge_index[0], edge_index[1]   # each (E,)
        E = src.shape[0]

        Wx = self.W_val(x)            # (B, N, H*D)
        Wx = Wx.view(B, N, H, D)     # (B, N, H, D)

        x_src = Wx[:, src, :, :]     # (B, E, H, D)
        x_dst = Wx[:, dst, :, :]     # (B, E, H, D)

        # Attention coefficient: LeakyReLU(a^T [x_src || x_dst])
        cat = torch.cat([x_src, x_dst], dim=-1)      # (B, E, H, 2D)
        # a: (H, 2D) → score: (B, E, H)
        score = (cat * self.a.unsqueeze(0).unsqueeze(0)).sum(-1)  # (B, E, H)
        score = F.leaky_relu(score, 0.2)

        # Incorporate signed edge weights: scale score by abs(weight), sign already in weight
        ew = edge_weight.abs().to(x.device)          # (E,) non-negative scaling
        score = score * ew.unsqueeze(0).unsqueeze(-1) # (B, E, H)

        # Softmax over incoming edges for each destination node
        # Use scatter-based softmax approximation via node-wise maximum
        out = torch.zeros(B, N, H, D, device=x.device)
        for h in range(H):
            s_h = score[:, :, h]    # (B, E)
            # Compute softmax denominator per destination node
            # Simple but correct: compute exp(s) / sum(exp(s)) per dst
            s_exp = torch.exp(s_h - s_h.max(dim=1, keepdim=True).values)  # (B, E)
            # Accumulate per dst
            denom = torch.zeros(B, N, device=x.device)
            denom.scatter_add_(1, dst.unsqueeze(0).expand(B, -1), s_exp)
            denom = denom.clamp(min=1e-8)
            alpha = s_exp / denom[:, dst]            # (B, E) normalised
            alpha = self.dropout(alpha)
            # Weighted sum of value vectors
            msg = alpha.unsqueeze(-1) * Wx[:, src, h, :]  # (B, E, D)
            out[:, :, h, :].scatter_add_(
                1,
                dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, D),
                msg
            )

        out = out.view(B, N, H * D)  # (B, N, H*D)
        # BN over (B*N, H*D)
        out = self.bn(out.view(B * N, H * D)).view(B, N, H * D)
        return F.elu(out)


class EuclideanTriModalGNN(nn.Module):
    """
    Euclidean counterpart of HyperLorentzNetHGCN.

    Stage 1: Per-modality TCN encoder → latent_dim Euclidean features per node.
             (Same as LEGEND Stage 1 WITHOUT LorentzProjection.)
    Stage 2: Euclidean GAT over PLV graph → mean pooling → classifier.
             (Same as LEGEND Stage 2 WITHOUT Lorentz/Fréchet geometry.)

    The only architectural difference from HyperLorentzNetHGCN is:
      - No LorentzProjection (nodes stay in R^latent_dim, not ℍ^latent_dim)
      - EuclideanGATLayer instead of LorentzGraphAttentionLayer
      - Mean pooling instead of Fréchet mean
    """

    def __init__(self, eeg_channels, esg_channels, emg_channels,
                 hidden_dim=64, latent_dim=32, num_classes=4,
                 gnn_hidden=32, gnn_layers=1, gnn_heads=1,
                 dropout=0.2, t_stride=4):
        super().__init__()
        self.t_stride = t_stride
        self.n_nodes  = eeg_channels + esg_channels + emg_channels
        self.num_classes = num_classes

        # Stage 1: per-channel TCN encoders (one encoder per modality, shared across channels)
        self.eeg_enc = ModalityEncoder(eeg_channels, hidden_dim, latent_dim, dropout)
        self.esg_enc = ModalityEncoder(esg_channels, hidden_dim, latent_dim, dropout)
        self.emg_enc = ModalityEncoder(emg_channels, hidden_dim, latent_dim, dropout)

        # Stage 1 classifier (auxiliary, same as LEGEND)
        self.stage1_classifier = nn.Linear(latent_dim * 3, num_classes)

        # Stage 2: Euclidean GAT layers
        in_dim = latent_dim
        gat_layers = []
        for i in range(gnn_layers):
            out = gnn_hidden if i < gnn_layers - 1 else gnn_hidden
            gat_layers.append(EuclideanGATLayer(in_dim, out, heads=gnn_heads, dropout=dropout))
            in_dim = out * gnn_heads
        self.gat_layers = nn.ModuleList(gat_layers)

        # Stage 2 classifier
        self.stage2_classifier = nn.Linear(in_dim, num_classes)

        # Graph buffers
        self.register_buffer("edge_index",  torch.zeros(2, 0, dtype=torch.long))
        self.register_buffer("edge_weight", torch.zeros(0))
        self._graph_registered = False

    def register_graph(self, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        self.edge_index  = edge_index
        self.edge_weight = edge_weight
        self._graph_registered = True
        print(f"[EuclideanTriModalGNN] Graph registered: {self.n_nodes} nodes, "
              f"{edge_index.shape[1]} directed edges")

    def forward(self, eeg, esg, emg):
        """
        eeg: (B, C_eeg, T)
        esg: (B, C_esg, T)
        emg: (B, C_emg, T)
        Returns dict with 'logits', 'stage1_logits'
        """
        # Optional temporal downsampling (same as LEGEND)
        if self.t_stride > 1:
            eeg = eeg[:, :, ::self.t_stride]
            esg = esg[:, :, ::self.t_stride]
            emg = emg[:, :, ::self.t_stride]

        # Stage 1: modality-level embeddings (B, latent_dim) each
        eeg_z = self.eeg_enc(eeg)
        esg_z = self.esg_enc(esg)
        emg_z = self.emg_enc(emg)

        stage1_feat = torch.cat([eeg_z, esg_z, emg_z], dim=-1)  # (B, 3*latent_dim)
        stage1_logits = self.stage1_classifier(stage1_feat)

        # Stage 2: GNN over per-node features
        # Treat each modality embedding as representing all its channels uniformly
        # (same node-feature initialisation as LEGEND)
        B = eeg.shape[0]
        eeg_nodes = eeg_z.unsqueeze(1).expand(-1, self.eeg_enc.proj.out_features
                                              if hasattr(self.eeg_enc.proj, 'out_features')
                                              else eeg_z.shape[-1], -1)
        # Each channel in a modality gets the same embedding -> (B, C_mod, latent_dim)
        n_eeg = eeg.shape[1] // self.t_stride if self.t_stride > 0 else eeg.shape[1]
        # Expand modality embeddings to per-channel node features
        eeg_n = eeg_z.unsqueeze(1).expand(-1, N_EEG, -1)   # (B, 28, latent_dim)
        esg_n = esg_z.unsqueeze(1).expand(-1, N_ESG, -1)   # (B, 15, latent_dim)
        emg_n = emg_z.unsqueeze(1).expand(-1, N_EMG, -1)   # (B,  8, latent_dim)
        node_feats = torch.cat([eeg_n, esg_n, emg_n], dim=1)  # (B, 51, latent_dim)

        # Apply GAT layers
        x = node_feats
        ei = self.edge_index.to(eeg.device)
        ew = self.edge_weight.to(eeg.device)
        for layer in self.gat_layers:
            x = layer(x, ei, ew)

        # Mean pooling over all nodes
        trial_embed = x.mean(dim=1)   # (B, gnn_hidden * heads)
        logits = self.stage2_classifier(trial_embed)

        # Ensemble with Stage 1 (same 0.5/0.5 mix as LEGEND)
        combined = 0.5 * logits + 0.5 * stage1_logits

        return {"logits": combined, "stage1_logits": stage1_logits}


# ---------------------------------------------------------------------------
# One LOSO fold
# ---------------------------------------------------------------------------

def train_fold(test_subj, subjects, data_dir, results_dir, args, device):
    print(f"\n{'='*60}")
    print(f"FOLD: hold-out = {test_subj}")
    print(f"{'='*60}")

    train_subjects = [s for s in subjects if s != test_subj]
    train_parts = [load_subject(data_dir, s) for s in train_subjects]
    labels_tr = np.concatenate([p[3] for p in train_parts])

    eeg_te, esg_te, emg_te, labels_te = load_subject(data_dir, test_subj)

    num_classes  = len(np.unique(labels_tr))
    test_classes = sorted(np.unique(labels_te).tolist())
    n_train = sum(len(p[3]) for p in train_parts)
    print(f"  Train: {n_train} trials | Test: {eeg_te.shape[0]} trials | Classes (train): {num_classes}")
    print(f"  Classes present in test subject: {test_classes}")

    # PLV graph (R2.d fix: class-stratified)
    print("  Building tri-layer PLV graph (stratified) ...")
    t0 = time.time()
    builder = TriLayerGraphBuilder(N_EEG, N_ESG, N_EMG)
    eeg_g, esg_g, emg_g = stratified_sample_plv(train_parts, n_total=256, seed=args.seed)
    edge_index, edge_weight = builder.build(eeg_g, esg_g, emg_g,
                                            k_cross=args.k_cross, signed=True)
    del eeg_g, esg_g, emg_g
    print(f"  Graph built in {time.time()-t0:.1f}s | {edge_index.shape[1]} edges")

    # Model
    model = EuclideanTriModalGNN(
        eeg_channels=N_EEG, esg_channels=N_ESG, emg_channels=N_EMG,
        hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
        num_classes=num_classes,
        gnn_hidden=args.gnn_hidden, gnn_layers=args.gnn_layers,
        gnn_heads=args.gnn_heads, dropout=args.dropout,
        t_stride=args.t_stride,
    )
    model.register_graph(edge_index, edge_weight)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")

    # Optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    # Class-weighted loss (R2.d fix)
    class_counts = np.bincount(labels_tr, minlength=num_classes).astype(np.float32)
    class_counts = np.where(class_counts == 0, 1e6, class_counts)
    class_weights = torch.tensor(1.0 / (class_counts + 1e-6)).to(device)
    class_weights = class_weights / class_weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"  Class counts (train): {class_counts.astype(int).tolist()}")

    # Loaders (R2.b fix: trial-level stratified 20% val, never use test subject)
    tr_splits, val_splits = [], []
    for eeg_s, esg_s, emg_s, lab_s in train_parts:
        split = stratified_val_split(eeg_s, esg_s, emg_s, lab_s, val_frac=0.20, seed=args.seed)
        tr_splits.append(split[:4])
        val_splits.append(split[4:])

    train_loader = DataLoader(
        ConcatDataset([AugDataset(t[0], t[1], t[2], t[3], augment=True) for t in tr_splits]),
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    val_eeg = np.concatenate([v[0] for v in val_splits])
    val_esg = np.concatenate([v[1] for v in val_splits])
    val_emg = np.concatenate([v[2] for v in val_splits])
    val_lab = np.concatenate([v[3] for v in val_splits])
    val_loader  = make_loader(val_eeg, val_esg, val_emg, val_lab,
                              args.batch_size, shuffle=False, augment=False)
    test_loader = make_loader(eeg_te, esg_te, emg_te, labels_te,
                              args.batch_size, shuffle=False, augment=False)
    del val_eeg, val_esg, val_emg, val_lab

    # Training loop
    best_val_acc, best_state, patience_ctr = 0.0, None, 0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            eeg_b, esg_b, emg_b, lab_b = [t.to(device) for t in batch]
            optimizer.zero_grad()
            out = model(eeg_b, esg_b, emg_b)
            loss = criterion(out["logits"], lab_b)
            loss += 0.2 * criterion(out["stage1_logits"], lab_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct = total = 0
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

    # Test evaluation (best-val model, single evaluation, with absent-class masking)
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    absent_classes = [c for c in range(num_classes) if c not in test_classes]
    correct = total = 0
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

    # Save
    fold_dir = os.path.join(results_dir, f"fold_{test_subj}")
    os.makedirs(fold_dir, exist_ok=True)
    torch.save({"model_state": best_state,
                "edge_index": edge_index.cpu(),
                "edge_weight": edge_weight.cpu()},
               os.path.join(fold_dir, "best_model.pt"))

    fold_result = {
        "subject": test_subj,
        "best_acc": test_acc,
        "best_val_acc": best_val_acc,
        "history": history,
        "n_params": total_params,
        "n_edges": edge_index.shape[1],
        "model": "EuclideanTriModalGNN",
    }
    with open(os.path.join(fold_dir, "fold_result.json"), "w") as f:
        json.dump(fold_result, f, indent=2)

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
        args.device = "cpu"
        print("CUDA not available, falling back to CPU")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        print(f"Device: {device}  ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Device: {device}")

    print(f"Model: EuclideanTriModalGNN (R1.3 ablation)")
    print(f"Epochs={args.epochs}  Patience={args.patience}")
    os.makedirs(args.results_dir, exist_ok=True)

    fold_results = []
    for subj in SUBJECTS:
        result = train_fold(
            test_subj=subj,
            subjects=SUBJECTS,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            args=args,
            device=device,
        )
        fold_results.append(result)

    accs = [r["best_acc"] for r in fold_results]
    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs))

    print(f"\n{'='*60}")
    print(f"EuclideanTriModalGNN — LOSO Summary")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  {r['subject']}:  {r['best_acc']:.2f}%")
    print(f"\n  Mean: {mean_acc:.2f}%  Std: {std_acc:.2f}%")

    summary = {
        "model": "EuclideanTriModalGNN",
        "mean_acc": mean_acc,
        "std_acc":  std_acc,
        "per_subject": {r["subject"]: r["best_acc"] for r in fold_results},
        "args": vars(args),
    }
    with open(os.path.join(args.results_dir, "loso_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.results_dir}/loso_summary.json")


if __name__ == "__main__":
    main()
