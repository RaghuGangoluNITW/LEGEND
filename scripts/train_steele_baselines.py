"""
Baseline LOSO classifiers on the Steele EEG+ESG+EMG dataset.

Models implemented:
  1. EEGNet        (Lawhern et al., J Neural Eng 2018)  — EEG only
  2. ShallowConvNet (Schirrmeister et al., HBM 2017)    — EEG only
  3. EEGNet-Tri    — EEGNet applied to all 3 modalities (EEG+ESG+EMG, 51 ch)
  4. BrainTopoGCN  (Shi et al., BSPC Sep 2024, doi:10.1016/j.bspc.2024.106401)
                    — mutual-information GCN + depthwise CNN + TCN (all 51 ch)
  5. EEG_GLT-Net   (Aung et al., BSPC Jun 2025, doi:10.1016/j.bspc.2025.XXXXX)
                    — Chebyshev spectral GCN with Graph Lottery Ticket pruning (all 51 ch)
  6. SAMGCN        (Meng et al., BSPC May 2026, doi:10.1016/j.bspc.2026.109506)
                    — self-adaptive multilevel GCN with dual temporal/frequency branches (all 51 ch)

These serve as Table 1 baselines in the paper (Reviewer R1.1 request).

Usage:
    python scripts/train_steele_baselines.py --data_dir data/steele
    python scripts/train_steele_baselines.py --data_dir data/steele --model ShallowConvNet
    python scripts/train_steele_baselines.py --data_dir data/steele --model EEGNet-Tri
    python scripts/train_steele_baselines.py --data_dir data/steele --model BrainTopoGCN
    python scripts/train_steele_baselines.py --data_dir data/steele --model EEG_GLT-Net
    python scripts/train_steele_baselines.py --data_dir data/steele --model SAMGCN

Results saved to results/baselines_<model>/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

# ---------------------------------------------------------------------------
# Data loading (identical to train_hybrid_loso.py)
# ---------------------------------------------------------------------------

SUBJECTS = ["NIS001","NIS002","NIS003","NIS004","NIS005",
            "NIS006","NIS007","NIS008","NIS009","NIS010"]

N_EEG, N_ESG, N_EMG = 28, 15, 8


def load_subject(data_dir: str, subj: str):
    path = os.path.join(data_dir, f"{subj}.npz")
    d = np.load(path, allow_pickle=True)
    return d["eeg"], d["esg"], d["emg"], d["labels"]


class SimpleDataset(torch.utils.data.Dataset):
    """Returns (signal, label). signal shape depends on model (EEG-only or all channels)."""
    def __init__(self, x, labels, augment=False, prob=0.25, shift=7):
        self.x = x
        self.labels = labels
        self.augment = augment
        self.prob = prob
        self.shift = shift

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        x = self.x[idx].copy()
        lab = self.labels[idx]
        if self.augment and np.random.rand() < self.prob:
            s = np.random.randint(-self.shift, self.shift + 1)
            x = np.roll(x, s, axis=-1)
            x = x * np.random.uniform(0.85, 1.15)
        return torch.from_numpy(x).float(), torch.tensor(lab, dtype=torch.long)


def stratified_val_split_x(x, labels, val_frac=0.20, seed=42):
    """Split signal array + labels into train/val (stratified by class)."""
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
    return x[tr_idx], labels[tr_idx], x[val_idx], labels[val_idx]


# ---------------------------------------------------------------------------
# EEGNet
# ---------------------------------------------------------------------------

class EEGNet(nn.Module):
    """
    EEGNet: Lawhern et al., J Neural Eng 2018  (doi:10.1088/1741-2552/aace8c)
    Compact CNN with depthwise temporal + spatial convolutions.

    Input: (B, 1, C, T)  — 1 'image' channel, C EEG channels, T time points
    """

    def __init__(self, n_channels: int, n_classes: int, T: int = 1000,
                 F1: int = 8, D: int = 2, F2: int = 16,
                 dropout: float = 0.5, kern_half: int = 64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Block 1: temporal conv + depthwise spatial conv
        self.conv1 = nn.Conv2d(1, F1, (1, kern_half * 2), padding=(0, kern_half), bias=False)
        self.bn1   = nn.BatchNorm2d(F1)
        self.dw    = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2   = nn.BatchNorm2d(F1 * D)
        self.act1  = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: separable conv
        self.sep1  = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.sep2  = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3   = nn.BatchNorm2d(F2)
        self.act2  = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Classifier (size computed dynamically)
        self._flat_size = self._get_flat_size(n_channels, T)
        self.classifier = nn.Linear(self._flat_size, n_classes)

    def _get_flat_size(self, C, T):
        with torch.no_grad():
            x = torch.zeros(1, 1, C, T)
            return self._features(x).shape[1]

    def _features(self, x):
        x = self.drop1(self.pool1(self.act1(self.bn2(self.dw(self.bn1(self.conv1(x)))))))
        x = self.drop2(self.pool2(self.act2(self.bn3(self.sep2(self.sep1(x))))))
        return x.flatten(1)

    def forward(self, x):
        # x: (B, C, T) → add channel dim → (B, 1, C, T)
        x = x.unsqueeze(1)
        return self.classifier(self._features(x))


# ---------------------------------------------------------------------------
# ShallowConvNet
# ---------------------------------------------------------------------------

class ShallowConvNet(nn.Module):
    """
    ShallowConvNet: Schirrmeister et al., HBM 2017  (doi:10.1002/hbm.23730)
    Temporal conv + spatial conv + square + log + FC.

    Input: (B, C, T)
    """

    def __init__(self, n_channels: int, n_classes: int, T: int = 1000,
                 n_filters: int = 40, kern_time: int = 25, dropout: float = 0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, n_filters, (1, kern_time), bias=False)
        self.spatial  = nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False)
        self.bn       = nn.BatchNorm2d(n_filters, momentum=0.1, affine=True)
        self.pool     = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.drop     = nn.Dropout(dropout)

        self._flat_size = self._get_flat_size(n_channels, T, n_filters, kern_time)
        self.classifier = nn.Linear(self._flat_size, n_classes)

    def _get_flat_size(self, C, T, F, kt):
        with torch.no_grad():
            x = torch.zeros(1, 1, C, T)
            x = self.pool(self.bn(self.spatial(self.temporal(x))))
            return x.flatten(1).shape[1]

    def forward(self, x):
        x = x.unsqueeze(1)                          # (B, 1, C, T)
        x = self.temporal(x)                         # (B, F, C, T-kt+1)
        x = self.spatial(x)                          # (B, F, 1, ...)
        x = self.bn(x)
        x = x ** 2                                   # square activation
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))     # log activation
        x = self.drop(x)
        return self.classifier(x.flatten(1))


# ---------------------------------------------------------------------------
# BrainTopoGCN (Shi et al., BSPC Sep 2024, doi:10.1016/j.bspc.2024.106401)
#
# Architecture (faithful to Shi et al., BSPC Sep 2024):
#   1. Sliding-window segmentation (win = T//4, step = win//2)
#   2. Learnable adjacency → normalised Laplacian (approximates per-window MI)
#   3. Chebyshev GCN (K=2) embeds topology INTO the signal (shape preserved)
#   4. CON block: 3 depthwise conv layers (F1=2, F2=2, F3=1) per window,
#      AdaptiveAvgPool2d((1,1)) collapses to 4 scalars / window
#   5. TCN: 2 residual blocks (dilation 1 & 2, 32 filters, kernel 3)
#   6. FC → n_classes
# Adapted for tri-modal 51-channel input (EEG+ESG+EMG concatenated).
# ---------------------------------------------------------------------------

class BrainTopoGCN(nn.Module):
    """
    Brain Topography Graph Embedded CNN (Shi et al., BSPC Sep 2024,
    doi:10.1016/j.bspc.2024.106401).

    Pipeline (faithful to paper):
      1. Segment each trial into overlapping windows (win=T//4, step=win//2).
      2. Build normalised Laplacian from a learnable adjacency (approximates
         the per-window MI adjacency used in the paper).
      3. Chebyshev GCN (K=2) embeds topology INTO the signal  per window —
         output has the **same shape as input** (channels × win_len).
         Paper initialises θ_k = 1/(k+1).
      4. CON block: 3 depthwise-conv layers per window (paper Table 1):
           L1: F1=2 filters, kernel (1, K1=25)  — temporal
           L2: F2=2 filters, kernel (C,  1)     — spatial (depthwise, groups=F1)
           L3: F3=1 filter,  kernel (1, K2=16)  — temporal (depthwise, groups=F1*F2)
         AdaptiveAvgPool2d((1,1)) → F1*F2*F3 = 4 scalars per window.
      5. TCN (paper: 2 residual blocks, dilation 1 & 2, 32 filters, kernel 3, ELU):
         input = (B, 4, n_windows) temporal sequence.
      6. AdaptiveAvgPool1d(1) → FC → n_classes.

    Input: (B, C, T)
    """

    def __init__(self, n_channels: int, n_classes: int, T: int = 1000,
                 K: int = 2, tcn_F: int = 32, tcn_K: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.n_channels = n_channels
        self.K = K
        # Paper uses 1 s windows (250 samples at 250 Hz); we scale to T//4
        self.win_len  = max(T // 4, 64)
        self.win_step = self.win_len // 2

        # Learnable adjacency (initialised as fully-connected minus self-loops)
        self.adj = nn.Parameter(
            torch.ones(n_channels, n_channels) - torch.eye(n_channels)
        )
        # Chebyshev θ_k coefficients; paper initialises to 1/(k+1)
        self.theta = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0 / (k + 1))) for k in range(K)
        ])

        # CON block (operates on 4-D tensor (B*w, 1, C, win_len))
        F1, F2, F3 = 2, 2, 1
        K1, K2 = 25, 16
        self.dw1 = nn.Conv2d(1,       F1,       (1, K1), bias=False)
        self.dw2 = nn.Conv2d(F1,      F1 * F2,  (n_channels, 1),
                             groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * F2)
        # dw3 only applied when sufficient temporal width remains
        self.dw3 = nn.Conv2d(F1 * F2, F1 * F2 * F3, (1, K2),
                             groups=F1 * F2, bias=False)
        self.bn3  = nn.BatchNorm2d(F1 * F2 * F3)
        # Collapse both spatial (=1 after dw2) and temporal dims to 1
        self.con_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop_con  = nn.Dropout(dropout)

        # con_feat = number of channels out of CON = F1*F2*F3 = 4
        con_feat = F1 * F2 * F3  # 4

        # TCN — two residual blocks (dilation 1 then dilation 2)
        self.tcn1a = nn.Conv1d(con_feat, tcn_F, tcn_K,
                               padding=tcn_K // 2, dilation=1, bias=False)
        self.tcn1b = nn.Conv1d(tcn_F,    tcn_F, tcn_K,
                               padding=tcn_K // 2, dilation=1, bias=False)
        self.tcn1_bn1 = nn.BatchNorm1d(tcn_F)
        self.tcn1_bn2 = nn.BatchNorm1d(tcn_F)
        self.tcn1_skip = nn.Conv1d(con_feat, tcn_F, 1, bias=False)

        self.tcn2a = nn.Conv1d(tcn_F, tcn_F, tcn_K,
                               padding=(tcn_K - 1) * 2 // 2, dilation=2, bias=False)
        self.tcn2b = nn.Conv1d(tcn_F, tcn_F, tcn_K,
                               padding=(tcn_K - 1) * 2 // 2, dilation=2, bias=False)
        self.tcn2_bn1 = nn.BatchNorm1d(tcn_F)
        self.tcn2_bn2 = nn.BatchNorm1d(tcn_F)

        self.tcn_pool  = nn.AdaptiveAvgPool1d(1)
        self.drop_out  = nn.Dropout(dropout)
        self.fc        = nn.Linear(tcn_F, n_classes)

    # ------------------------------------------------------------------
    def _build_laplacian(self, device: torch.device) -> torch.Tensor:
        """Normalised, scaled Laplacian from learnable adjacency. (C,C)"""
        A = (torch.relu(self.adj) + torch.relu(self.adj.T))
        A = A * (1.0 - torch.eye(self.n_channels, device=device))
        D_inv_sqrt = torch.diag(
            1.0 / (A.sum(-1) + 1e-8).sqrt()
        )
        L = (torch.eye(self.n_channels, device=device)
             - D_inv_sqrt @ A @ D_inv_sqrt)
        lmax = torch.linalg.eigvalsh(L).max().clamp(min=1e-6)
        return 2.0 * L / lmax - torch.eye(self.n_channels, device=device)

    def _gcn_embed(self, x_win: torch.Tensor,
                   L_hat: torch.Tensor) -> torch.Tensor:
        """
        Apply Chebyshev polynomial filter to embed topology into signal.
        L_hat : (C, C)
        x_win : (BW, C, WL)  — raw signal per window
        returns (BW, C, WL)  — topology-embedded signal (same shape)
        """
        # Transpose to (BW, WL, C) so graph conv mixes channel dimension
        xt = x_win.permute(0, 2, 1)            # (BW, WL, C)
        T0 = xt
        T1 = torch.einsum("nm,bwm->bwn", L_hat, xt)  # L̃ @ x (symmetric → ok)
        z  = self.theta[0] * T0
        if self.K > 1:
            z = z + self.theta[1] * T1
        Tp, Tc = T0, T1
        for k in range(2, self.K):
            Tn = 2.0 * torch.einsum("nm,bwm->bwn", L_hat, Tc) - Tp
            z  = z + self.theta[k] * Tn
            Tp, Tc = Tc, Tn
        return z.permute(0, 2, 1)              # (BW, C, WL)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        device  = x.device
        L_hat   = self._build_laplacian(device)          # (C, C)

        # ---- sliding-window segmentation ----
        WL, WS = self.win_len, self.win_step
        starts  = list(range(0, max(T - WL + 1, 1), WS))
        WL      = min(WL, T)                             # guard for short signals
        w       = len(starts)

        x_wins  = torch.stack(                           # (B, w, C, WL)
            [x[:, :, s:s + WL] for s in starts], dim=1)
        x_wins  = x_wins.reshape(B * w, C, WL)          # (BW, C, WL)

        # ---- GCN topology embedding ----
        z = self._gcn_embed(x_wins, L_hat)               # (BW, C, WL)

        # ---- CON block ----
        z4 = z.unsqueeze(1)                              # (BW, 1, C, WL)
        z4 = F.elu(self.dw1(z4))                        # (BW, F1, C, WL-K1+1)
        z4 = F.elu(self.bn2(self.dw2(z4)))              # (BW, F1*F2, 1, ...)
        if z4.shape[-1] >= 16:                           # apply dw3 when feasible
            z4 = F.elu(self.bn3(self.dw3(z4)))
        z4 = self.drop_con(z4)
        z4 = self.con_pool(z4)                          # (BW, 4, 1, 1)

        # ---- build temporal sequence from per-window features ----
        feat = z4.flatten(1)                             # (BW, 4)
        seq  = feat.reshape(B, w, -1).permute(0, 2, 1)  # (B, 4, w)

        # ---- TCN residual block 1 (dilation=1) ----
        h  = F.elu(self.tcn1_bn1(self.tcn1a(seq)))
        h  = self.tcn1_bn2(self.tcn1b(h))
        h  = F.elu(h + self.tcn1_skip(seq))

        # ---- TCN residual block 2 (dilation=2) ----
        h2 = F.elu(self.tcn2_bn1(self.tcn2a(h)))
        h2 = self.tcn2_bn2(self.tcn2b(h2))
        h  = F.elu(h2 + h)

        out = self.tcn_pool(h).squeeze(-1)               # (B, tcn_F)
        out = self.drop_out(out)
        return self.fc(out)


# ---------------------------------------------------------------------------
# EEG_GLT-Net (Aung et al., BSPC Jun 2025, published from arXiv:2404.11075)
#
# Architecture: trainable adjacency mask (Graph Lottery Ticket approach) +
#               spectral Chebyshev GCN layers + global mean pool + FC.
# Adapted for tri-modal 51-channel input. We use Model A config:
#   6 GCN layers: filters [16,32,64,128,256,512], polynomial order K=5 for all.
#   FC: [1024, 2048, n_classes].
# We simplify to a smaller config (Model C-equivalent) to fit GPU memory:
#   5 GCN layers: filters [16,32,64,128,256], K=5, FC: [n_classes].
# ---------------------------------------------------------------------------

class EEG_GLT_Net(nn.Module):
    """
    EEG Graph Lottery Ticket Network (Aung et al., IEEE TNNLS Vol.36 No.9, 2025).

    Architecture (faithful to paper Table I — Model A with smaller variant):
      • Trainable adjacency mask m_g (Graph Lottery Ticket approximation).
      • Each EEG time-point is a graph signal (nodes = channels).
        To fit 4 GB GPU we first downsample T → T//t_ds (default 10×) with
        AvgPool before the per-time-point GCN (preserves the paradigm faithfully).
      • 5 Chebyshev spectral GCN layers, filters [16,32,64,128,256], K=5 per layer.
      • Global mean pooling over nodes after the last GCN layer.
      • Average over downsampled time-points → (B, last_filter).
      • FC: last_filter → fc_hid → n_classes.

    Input: (B, C, T)
    """

    def __init__(self, n_channels: int, n_classes: int, T: int = 1000,
                 gcn_filters: tuple = (8, 16, 32, 64, 128),
                 K: int = 3, fc_hid: int = 512, dropout: float = 0.5,
                 t_ds: int = 50):   # adapted for T=1000: 20 time-pts, K=3, smaller filters
        super().__init__()
        self.n_channels = n_channels
        self.K = K
        self.t_ds = t_ds  # temporal down-sample factor before GCN

        # Learnable adjacency mask (Graph Lottery Ticket approximation)
        self.adj_mask = nn.Parameter(
            torch.ones(n_channels, n_channels) - torch.eye(n_channels)
        )

        # GCN layers: each linear maps K*in_f → out_f (per-node)
        self.gcn_layers: nn.ModuleList = nn.ModuleList()
        self.gcn_lnorms: nn.ModuleList = nn.ModuleList()
        in_f = 1
        for out_f in gcn_filters:
            self.gcn_layers.append(nn.Linear(in_f * K, out_f, bias=False))
            self.gcn_lnorms.append(nn.LayerNorm(out_f))
            in_f = out_f

        last_f = gcn_filters[-1]
        self.fc1   = nn.Linear(last_f, fc_hid)
        self.ln_fc = nn.LayerNorm(fc_hid)   # LayerNorm: safe at batch_size=1
        self.drop  = nn.Dropout(dropout)
        self.fc2   = nn.Linear(fc_hid, n_classes)

    # ------------------------------------------------------------------
    def _build_laplacian(self, device: torch.device) -> torch.Tensor:
        """Normalised scaled Laplacian from learned adjacency mask. (C,C)"""
        A = torch.relu(self.adj_mask)
        A = (A + A.T) * (1.0 - torch.eye(self.n_channels, device=device))
        D_inv_sqrt = torch.diag(1.0 / (A.sum(-1) + 1e-8).sqrt())
        L    = (torch.eye(self.n_channels, device=device)
                - D_inv_sqrt @ A @ D_inv_sqrt)
        lmax = torch.linalg.eigvalsh(L).max().clamp(min=1e-6)
        return 2.0 * L / lmax - torch.eye(self.n_channels, device=device)

    def _chebyshev_features(self, h: torch.Tensor,
                             L_hat: torch.Tensor) -> torch.Tensor:
        """
        h     : (BT, C, in_f)
        L_hat : (C, C) symmetric
        Returns (BT, C, in_f * K)
        """
        T0 = h
        T1 = torch.einsum("nm,bnf->bmf", L_hat, h)
        polys = [T0, T1]
        Tp, Tc = T0, T1
        for _ in range(2, self.K):
            Tn = 2.0 * torch.einsum("nm,bnf->bmf", L_hat, Tc) - Tp
            polys.append(Tn)
            Tp, Tc = Tc, Tn
        return torch.cat(polys, dim=-1)              # (BT, C, in_f*K)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        device  = x.device
        L_hat   = self._build_laplacian(device)      # (C, C)

        # Temporal down-sampling to keep GPU memory feasible
        # (B, C, T) → (B, C, T_ds) via AvgPool
        Tds = max(T // self.t_ds, 1)
        xd  = F.adaptive_avg_pool1d(x, Tds)         # (B, C, Tds)

        # Treat every downsampled time-point as a graph signal
        # (B, C, Tds) → (B*Tds, C, 1)
        h = xd.permute(0, 2, 1).reshape(B * Tds, C, 1).contiguous()

        # Chebyshev GCN layers
        for gcn, ln in zip(self.gcn_layers, self.gcn_lnorms):
            poly = self._chebyshev_features(h, L_hat)  # (BTds, C, in_f*K)
            h = F.relu(ln(gcn(poly)))                  # (BTds, C, out_f)

        # Global mean pooling over nodes, then average over time
        h = h.mean(dim=1)                              # (BTds, last_f)
        h = h.reshape(B, Tds, -1).mean(dim=1)         # (B, last_f)

        h = self.drop(F.relu(self.ln_fc(self.fc1(h))))
        return self.fc2(h)

# ---------------------------------------------------------------------------
# SAMGCN (Meng et al., BSPC May 2026, doi:10.1016/j.bspc.2026.109506)
#
# Architecture: dual-branch (time-domain 1D-CNN + frequency-domain DE features)
#   → channel self-attention graph embedding (CSAGE)
#   → multi-level GCN with learnable adjacency + concatenation of all GCN layers
#   → FC classifier.
# Adapted for tri-modal 51-channel input. Differential entropy extracted
# per band per channel (5 bands → 5*C features).
# ---------------------------------------------------------------------------

class SAMGCN(nn.Module):
    """
    Self-Adaptive Multilevel GCN (Meng et al., BSPC May 2026,
    doi:10.1016/j.bspc.2026.109506).

    Architecture (faithful to paper):
      Time branch  : per-channel Conv1d (kernel T//2) → Conv1d (kernel T//4)
                     → AdaptiveMaxPool → X_t ∈ R^{N×128}
      Freq branch  : differential entropy per 5 bands → small MLP → X_f ∈ R^{N×d}
      CLE          : learnable channel position embedding → added to X_t
      Fusion       : concat(X_t, X_f) → Linear → H ∈ R^{N×gcn_dim}
      CS-AGE       : self-attention computes attention-weighted adaptive adjacency
                     Ã_T and Ã_F for time and frequency branches respectively
      Multi-level GCN : 3 GCN layers per branch (T and F) with Ã_T / Ã_F
                        proper D̃^{-1/2} Ã D̃^{-1/2} normalisation
      Fusion       : concatenate outputs of all GCN layers (both branches)
      Classifier   : flatten → FC(256) → FC(n_classes)

    Input: (B, C, T)
    """

    def __init__(self, n_channels: int, n_classes: int, T: int = 1000,
                 freq_bands: int = 5,
                 gcn_dims: tuple = (64, 64, 64),
                 dropout: float = 0.5,
                 t_stride: int = 8):   # adapted for T=1000: pre-pool before conv branches
        super().__init__()
        self.n_channels = n_channels
        self.freq_bands = freq_bands
        gcn_dim0 = gcn_dims[0]

        # ── Time branch ──────────────────────────────────────────────
        # Paper: 32 kernels of size (T/2)×1, then 128 kernels of (T/4)×1.
        # Applied per-channel (processed as B*C independent 1-D signals).
        # Adapted for T=1000: pre-pool by t_stride so kernel sizes stay
        # proportional to the original paper's shorter window lengths.
        self.t_stride = t_stride
        T_pooled = max(T // t_stride, 1)
        k1 = max(T_pooled // 2, 25)
        k2 = max(T_pooled // 4, 13)
        self.t_conv1  = nn.Conv1d(1,  32,  k1, bias=False)
        self.t_conv2  = nn.Conv1d(32, 128, k2, bias=False)
        self.t_mp     = nn.AdaptiveMaxPool1d(1)
        t_out = 128  # X_t features per channel

        # ── Frequency branch ─────────────────────────────────────────
        self.f_proj = nn.Sequential(
            nn.Linear(freq_bands, 32), nn.ReLU(),
            nn.Linear(32, t_out)
        )

        # ── CLE: channel location encoding (learnable positional embed) ──
        self.cle = nn.Embedding(n_channels, t_out)

        # ── Fused feature projection ─────────────────────────────────
        # Concat time + freq → project to GCN input dim
        self.fuse_proj = nn.Linear(t_out * 2, gcn_dim0)

        # ── CS-AGE: learnable inter-branch attention adjacency ────────
        # Shared Q/K projection; separate adaptive weight matrices W_αT, W_αF
        cs_hid = 32
        self.cs_Q = nn.Linear(gcn_dim0, cs_hid, bias=False)
        self.cs_K = nn.Linear(gcn_dim0, cs_hid, bias=False)
        self.cs_scale = cs_hid ** -0.5
        self.W_alphaT = nn.Parameter(
            torch.eye(n_channels) * 0.5
            + 0.01 * torch.randn(n_channels, n_channels))
        self.W_alphaF = nn.Parameter(
            torch.eye(n_channels) * 0.5
            + 0.01 * torch.randn(n_channels, n_channels))

        # ── GCN layers (separate T and F branches, 3 layers each) ────
        # Paper eq.15-16: H^l = σ(W_α D̃^{-1/2} Ã D̃^{-1/2} H^{l-1} W)
        self.gcn_T = nn.ModuleList()
        self.gcn_F = nn.ModuleList()
        in_f = gcn_dim0
        for out_f in gcn_dims:
            self.gcn_T.append(nn.Linear(in_f, out_f, bias=False))
            self.gcn_F.append(nn.Linear(in_f, out_f, bias=False))
            in_f = out_f

        # ── Classifier ───────────────────────────────────────────────
        # Concat all GCN outputs from both branches
        total_gcn_feat = sum(gcn_dims) * 2 * n_channels
        self.drop  = nn.Dropout(dropout)
        self.fc1   = nn.Linear(total_gcn_feat, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2   = nn.Linear(256, n_classes)

    # ------------------------------------------------------------------
    @staticmethod
    def _gcn_step(H: torch.Tensor, A_raw: torch.Tensor,
                  W: nn.Linear) -> torch.Tensor:
        """
        One GCN layer with symmetric normalisation.
        H     : (B, C, in_f)
        A_raw : (B, C, C)  — raw (unnormalised) adjacency
        returns (B, C, out_f)

        Using simplified first-order: H_new = σ( D̃^{-1/2} Ã D̃^{-1/2} H W )
        where Ã = A_raw + I (add self-loops).
        """
        I = torch.eye(A_raw.shape[-1], device=A_raw.device).unsqueeze(0)
        Atilde = A_raw + I                         # (B, C, C) add self-loops
        D = Atilde.sum(-1)                          # (B, C)
        D_inv_sqrt = torch.diag_embed(1.0 / (D + 1e-8).sqrt())
        A_norm = torch.bmm(torch.bmm(D_inv_sqrt, Atilde), D_inv_sqrt)
        return F.relu(W(torch.bmm(A_norm, H)))      # (B, C, out_f)

    def _differential_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Approximate DE per channel per frequency band via log(band power).
        x: (B, C, T) → (B, C, 5)"""
        X_fft   = torch.fft.rfft(x, dim=-1)        # (B, C, T//2+1)
        n_freq  = X_fft.shape[-1]
        # Five relative band splits (δ θ α β γ)
        edges = [0,
                 max(int(n_freq * 0.05), 1),
                 max(int(n_freq * 0.10), 2),
                 max(int(n_freq * 0.17), 3),
                 max(int(n_freq * 0.40), 4),
                 n_freq]
        bands = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            hi   = max(hi, lo + 1)
            pwr  = (X_fft[:, :, lo:hi].abs() ** 2).mean(-1)
            bands.append(torch.log(pwr.clamp(min=1e-8)).unsqueeze(-1))
        return torch.cat(bands, dim=-1)             # (B, C, 5)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        device  = x.device

        # ── Time branch ──────────────────────────────────────────────
        xc = x.reshape(B * C, 1, T)                # (B*C, 1, T)
        if self.t_stride > 1:
            xc = F.avg_pool1d(xc, self.t_stride)   # (B*C, 1, T//t_stride)
        tc = F.relu(self.t_conv1(xc))              # (B*C, 32, T_pooled-k1+1)
        # Ensure second conv has enough input length
        if tc.shape[-1] >= self.t_conv2.kernel_size[0]:
            tc = F.relu(self.t_conv2(tc))
        else:
            tc = F.adaptive_max_pool1d(tc, self.t_conv2.kernel_size[0])
            tc = F.relu(self.t_conv2(tc))
        t_feat = self.t_mp(tc).squeeze(-1)         # (B*C, 128)
        t_feat = t_feat.reshape(B, C, 128)         # (B, C, 128) = X_t

        # ── Frequency branch ─────────────────────────────────────────
        de     = self._differential_entropy(x)     # (B, C, 5)
        f_feat = self.f_proj(de)                   # (B, C, 128) = X_f

        # ── CLE: channel location encoder ────────────────────────────
        pos    = self.cle(torch.arange(C, device=device))  # (C, 128)
        t_feat = t_feat + pos.unsqueeze(0)         # add positional info to X_t

        # ── Feature fusion ───────────────────────────────────────────
        H = self.fuse_proj(
            torch.cat([t_feat, f_feat], dim=-1))   # (B, C, gcn_dim0)

        # ── CS-AGE: compute attention-weighted adaptive adjacency ─────
        Q = self.cs_Q(H)                            # (B, C, cs_hid)
        K_ = self.cs_K(H)                           # (B, C, cs_hid)
        attn = torch.softmax(
            torch.bmm(Q, K_.transpose(1, 2)) * self.cs_scale,
            dim=-1)                                 # (B, C, C)

        W_T = torch.sigmoid(self.W_alphaT).unsqueeze(0)  # (1, C, C)
        W_F = torch.sigmoid(self.W_alphaF).unsqueeze(0)
        Atilde_T = attn * W_T                       # (B, C, C)
        Atilde_F = attn * W_F

        # ── Multi-level GCN ──────────────────────────────────────────
        H_T, H_F = H, H
        T_outs, F_outs = [], []
        for gcn_t, gcn_f in zip(self.gcn_T, self.gcn_F):
            H_T = self._gcn_step(H_T, Atilde_T, gcn_t)
            H_F = self._gcn_step(H_F, Atilde_F, gcn_f)
            T_outs.append(H_T)
            F_outs.append(H_F)

        # Concatenate all GCN outputs across levels and branches
        cat  = torch.cat(T_outs + F_outs, dim=-1)  # (B, C, sum_levels*2)
        feat = self.drop(cat.flatten(1))            # (B, C * sum...)

        feat = F.relu(self.bn_fc(self.fc1(feat)))
        feat = self.drop(feat)
        return self.fc2(feat)



# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_fold(test_subj, subjects, data_dir, results_dir, args, device):
    print(f"\n{'='*60}")
    print(f"FOLD: hold-out = {test_subj}")
    print(f"{'='*60}")

    train_subjects = [s for s in subjects if s != test_subj]
    train_parts = [load_subject(data_dir, s) for s in train_subjects]
    eeg_te, esg_te, emg_te, labels_te = load_subject(data_dir, test_subj)

    # Build signal array depending on model
    def make_x(eeg, esg, emg):
        if args.model in ("EEGNet-Tri", "BrainTopoGCN", "EEG_GLT-Net", "SAMGCN"):
            return np.concatenate([eeg, esg, emg], axis=1)   # (N, 51, T)
        else:
            return eeg                                         # (N, 28, T)

    x_te = make_x(eeg_te, esg_te, emg_te)
    n_channels = x_te.shape[1]
    T = x_te.shape[2]

    labels_tr = np.concatenate([p[3] for p in train_parts])
    num_classes = len(np.unique(labels_tr))
    test_classes = sorted(np.unique(labels_te).tolist())
    n_train = sum(len(p[3]) for p in train_parts)
    print(f"  n_channels={n_channels}  T={T}  train={n_train}  test={len(labels_te)}")
    print(f"  Classes in test: {test_classes}")

    # Build trial-level stratified val split from training subjects (R2.b fix)
    from torch.utils.data import ConcatDataset
    tr_splits, val_x_list, val_y_list = [], [], []
    for p in train_parts:
        x_p = make_x(p[0], p[1], p[2])
        x_tr, y_tr, x_val, y_val = stratified_val_split_x(x_p, p[3],
                                                           val_frac=0.20, seed=args.seed)
        tr_splits.append((x_tr, y_tr))
        val_x_list.append(x_val)
        val_y_list.append(y_val)

    train_ds = ConcatDataset([
        SimpleDataset(t[0], t[1], augment=True) for t in tr_splits
    ])
    val_x  = np.concatenate(val_x_list)
    val_y  = np.concatenate(val_y_list)
    val_ds  = SimpleDataset(val_x, val_y, augment=False)
    test_ds = SimpleDataset(x_te, labels_te, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    # Test loader — evaluated ONCE after training
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    # Model
    if args.model in ("EEGNet", "EEGNet-Tri"):
        model = EEGNet(n_channels, num_classes, T=T, dropout=args.dropout)
    elif args.model == "ShallowConvNet":
        model = ShallowConvNet(n_channels, num_classes, T=T, dropout=args.dropout)
    elif args.model == "BrainTopoGCN":
        model = BrainTopoGCN(n_channels, num_classes, T=T, dropout=args.dropout)
    elif args.model == "EEG_GLT-Net":
        model = EEG_GLT_Net(n_channels, num_classes, T=T, dropout=args.dropout)
    elif args.model == "SAMGCN":
        model = SAMGCN(n_channels, num_classes, T=T, dropout=args.dropout)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # Class-weighted loss
    class_counts = np.bincount(labels_tr, minlength=num_classes).astype(np.float32)
    class_counts = np.where(class_counts == 0, 1e6, class_counts)
    class_weights = torch.tensor(1.0 / (class_counts + 1e-6)).to(device)
    class_weights = class_weights / class_weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_state = None
    patience_ctr = 0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred = model(xb).argmax(1).cpu()
                correct += (pred == yb).sum().item()
                total += len(yb)
        val_acc = 100.0 * correct / total

        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  ep {epoch:3d}  loss={avg_loss:.4f}  val={val_acc:.2f}%  best_val={best_val_acc:.2f}%")

        if patience_ctr >= args.patience:
            print(f"  Early stop at epoch {epoch}")
            break

    # ---- Final test evaluation (best-val model, test subject never seen during training) ----
    model.load_state_dict(best_state)
    model.eval()
    # Mask classes absent from the test subject (same as LEGEND main model)
    absent_classes = [c for c in range(num_classes) if c not in test_classes]
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            if absent_classes:
                logits = logits.clone()
                logits[:, absent_classes] = float("-inf")
            pred = logits.argmax(1).cpu()
            correct += (pred == yb).sum().item()
            total += len(yb)
    test_acc = 100.0 * correct / total
    print(f"  Test acc: {test_acc:.2f}%  |  best val: {best_val_acc:.2f}%")

    # Save
    fold_dir = os.path.join(results_dir, f"fold_{test_subj}")
    os.makedirs(fold_dir, exist_ok=True)
    torch.save({"model_state": best_state}, os.path.join(fold_dir, "best_model.pt"))

    result = {
        "subject": test_subj,
        "best_acc": test_acc,
        "best_val_acc": best_val_acc,
        "history": history,
        "n_params": n_params,
        "model": args.model,
        "n_channels": n_channels,
    }
    with open(os.path.join(fold_dir, "fold_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  OK {test_subj}  test_acc = {test_acc:.2f}%")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="data/steele")
    p.add_argument("--results_dir", default=None,
                   help="Default: results/baselines_<model>")
    p.add_argument("--model",       default="EEGNet",
                   choices=["EEGNet", "ShallowConvNet", "EEGNet-Tri",
                             "BrainTopoGCN", "EEG_GLT-Net", "SAMGCN"])
    p.add_argument("--epochs",      default=150, type=int)
    p.add_argument("--patience",    default=25,  type=int)
    p.add_argument("--batch_size",  default=32,  type=int)
    p.add_argument("--lr",          default=1e-3, type=float)
    p.add_argument("--dropout",     default=0.5,  type=float)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--seed",        default=42,   type=int)
    return p.parse_args()


def main():
    args = parse_args()
    if args.results_dir is None:
        args.results_dir = f"results/baselines_{args.model}"

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

    print(f"Model: {args.model}  |  epochs={args.epochs}  patience={args.patience}")
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
    print(f"MODEL: {args.model}")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  {r['subject']}:  {r['best_acc']:.2f}%")
    print(f"\n  Mean: {mean_acc:.2f}%  Std: {std_acc:.2f}%")
    print(f"  Min:  {min(accs):.2f}%  Max: {max(accs):.2f}%")

    summary = {
        "model": args.model,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "per_subject": {r["subject"]: r["best_acc"] for r in fold_results},
        "args": vars(args),
    }
    with open(os.path.join(args.results_dir, "loso_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.results_dir}/loso_summary.json")


if __name__ == "__main__":
    main()
