#!/usr/bin/env python3
"""
BCI Competition IV-2a — LOSO baselines (BrainTopoGCN, EEG-GLT-Net, SAMGCN)
============================================================================

Protocol: strict LOSO — train on 8 subjects (T+E combined), test on 1.
Uses EEG channels only (25 ch after EOG exclusion), 4-class MI, chance=25%.

Model classes copied verbatim from train_steele_baselines.py with only the
channel count adapted (25 EEG instead of 51 tri-modal).

Usage:
    python scripts/train_bciciv2a_baselines_loso.py --model BrainTopoGCN
    python scripts/train_bciciv2a_baselines_loso.py --model EEG_GLT-Net
    python scripts/train_bciciv2a_baselines_loso.py --model SAMGCN
    python scripts/train_bciciv2a_baselines_loso.py --model all
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import mne


# ===========================================================================
# Model definitions (identical to train_steele_baselines.py)
# ===========================================================================

class BrainTopoGCN(nn.Module):
    def __init__(self, n_channels, n_classes, T=500, K=2, tcn_F=32, tcn_K=3, dropout=0.3):
        super().__init__()
        self.n_channels = n_channels
        self.K = K
        self.win_len  = max(T // 4, 64)
        self.win_step = self.win_len // 2
        self.adj = nn.Parameter(
            torch.ones(n_channels, n_channels) - torch.eye(n_channels))
        self.theta = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0 / (k + 1))) for k in range(K)])
        F1, F2, F3 = 2, 2, 1
        K1, K2 = 25, 16
        self.dw1 = nn.Conv2d(1, F1, (1, K1), bias=False)
        self.dw2 = nn.Conv2d(F1, F1*F2, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1*F2)
        self.dw3 = nn.Conv2d(F1*F2, F1*F2*F3, (1, K2), groups=F1*F2, bias=False)
        self.bn3 = nn.BatchNorm2d(F1*F2*F3)
        self.con_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop_con = nn.Dropout(dropout)
        con_feat = F1*F2*F3
        self.tcn1a = nn.Conv1d(con_feat, tcn_F, tcn_K, padding=tcn_K//2, dilation=1, bias=False)
        self.tcn1b = nn.Conv1d(tcn_F, tcn_F, tcn_K, padding=tcn_K//2, dilation=1, bias=False)
        self.tcn1_bn1 = nn.BatchNorm1d(tcn_F)
        self.tcn1_bn2 = nn.BatchNorm1d(tcn_F)
        self.tcn1_skip = nn.Conv1d(con_feat, tcn_F, 1, bias=False)
        self.tcn2a = nn.Conv1d(tcn_F, tcn_F, tcn_K, padding=(tcn_K-1)*2//2, dilation=2, bias=False)
        self.tcn2b = nn.Conv1d(tcn_F, tcn_F, tcn_K, padding=(tcn_K-1)*2//2, dilation=2, bias=False)
        self.tcn2_bn1 = nn.BatchNorm1d(tcn_F)
        self.tcn2_bn2 = nn.BatchNorm1d(tcn_F)
        self.tcn_pool = nn.AdaptiveAvgPool1d(1)
        self.drop_out = nn.Dropout(dropout)
        self.fc = nn.Linear(tcn_F, n_classes)

    def _build_laplacian(self, device):
        A = (torch.relu(self.adj) + torch.relu(self.adj.T))
        A = A * (1.0 - torch.eye(self.n_channels, device=device))
        D_inv_sqrt = torch.diag(1.0 / (A.sum(-1) + 1e-8).sqrt())
        L = torch.eye(self.n_channels, device=device) - D_inv_sqrt @ A @ D_inv_sqrt
        lmax = torch.linalg.eigvalsh(L).max().clamp(min=1e-6)
        return 2.0*L/lmax - torch.eye(self.n_channels, device=device)

    def _gcn_embed(self, x_win, L_hat):
        xt = x_win.permute(0, 2, 1)
        T0 = xt
        T1 = torch.einsum("nm,bwm->bwn", L_hat, xt)
        z  = self.theta[0] * T0
        if self.K > 1:
            z = z + self.theta[1] * T1
        Tp, Tc = T0, T1
        for k in range(2, self.K):
            Tn = 2.0 * torch.einsum("nm,bwm->bwn", L_hat, Tc) - Tp
            z  = z + self.theta[k] * Tn
            Tp, Tc = Tc, Tn
        return z.permute(0, 2, 1)

    def forward(self, x):
        B, C, T = x.shape
        device  = x.device
        L_hat   = self._build_laplacian(device)
        WL, WS  = self.win_len, self.win_step
        starts  = list(range(0, max(T - WL + 1, 1), WS))
        WL      = min(WL, T)
        w       = len(starts)
        x_wins  = torch.stack([x[:, :, s:s+WL] for s in starts], dim=1)
        x_wins  = x_wins.reshape(B*w, C, WL)
        z       = self._gcn_embed(x_wins, L_hat)
        z4      = z.unsqueeze(1)
        z4      = F.elu(self.dw1(z4))
        z4      = F.elu(self.bn2(self.dw2(z4)))
        if z4.shape[-1] >= 16:
            z4  = F.elu(self.bn3(self.dw3(z4)))
        z4      = self.drop_con(z4)
        z4      = self.con_pool(z4)
        feat    = z4.flatten(1)
        seq     = feat.reshape(B, w, -1).permute(0, 2, 1)
        h       = F.elu(self.tcn1_bn1(self.tcn1a(seq)))
        h       = self.tcn1_bn2(self.tcn1b(h))
        h       = F.elu(h + self.tcn1_skip(seq))
        h2      = F.elu(self.tcn2_bn1(self.tcn2a(h)))
        h2      = self.tcn2_bn2(self.tcn2b(h2))
        h       = F.elu(h2 + h)
        out     = self.tcn_pool(h).squeeze(-1)
        out     = self.drop_out(out)
        return self.fc(out)


class EEG_GLT_Net(nn.Module):
    def __init__(self, n_channels, n_classes, T=500,
                 gcn_filters=(8,16,32,64,128), K=3, fc_hid=512,
                 dropout=0.5, t_ds=50):
        super().__init__()
        self.n_channels = n_channels
        self.K = K
        self.t_ds = t_ds
        self.adj_mask = nn.Parameter(
            torch.ones(n_channels, n_channels) - torch.eye(n_channels))
        self.gcn_layers = nn.ModuleList()
        self.gcn_lnorms = nn.ModuleList()
        in_f = 1
        for out_f in gcn_filters:
            self.gcn_layers.append(nn.Linear(in_f*K, out_f, bias=False))
            self.gcn_lnorms.append(nn.LayerNorm(out_f))
            in_f = out_f
        last_f = gcn_filters[-1]
        self.fc1   = nn.Linear(last_f, fc_hid)
        self.ln_fc = nn.LayerNorm(fc_hid)
        self.drop  = nn.Dropout(dropout)
        self.fc2   = nn.Linear(fc_hid, n_classes)

    def _build_laplacian(self, device):
        A = torch.relu(self.adj_mask)
        A = (A + A.T) * (1.0 - torch.eye(self.n_channels, device=device))
        D_inv_sqrt = torch.diag(1.0 / (A.sum(-1) + 1e-8).sqrt())
        L = torch.eye(self.n_channels, device=device) - D_inv_sqrt @ A @ D_inv_sqrt
        lmax = torch.linalg.eigvalsh(L).max().clamp(min=1e-6)
        return 2.0*L/lmax - torch.eye(self.n_channels, device=device)

    def _chebyshev_features(self, h, L_hat):
        T0 = h; T1 = torch.einsum("nm,bnf->bmf", L_hat, h)
        polys = [T0, T1]; Tp, Tc = T0, T1
        for _ in range(2, self.K):
            Tn = 2.0*torch.einsum("nm,bnf->bmf", L_hat, Tc) - Tp
            polys.append(Tn); Tp, Tc = Tc, Tn
        return torch.cat(polys, dim=-1)

    def forward(self, x):
        B, C, T = x.shape
        device  = x.device
        L_hat   = self._build_laplacian(device)
        Tds     = max(T // self.t_ds, 1)
        xd      = F.adaptive_avg_pool1d(x, Tds)
        h       = xd.permute(0, 2, 1).reshape(B*Tds, C, 1).contiguous()
        for gcn, ln in zip(self.gcn_layers, self.gcn_lnorms):
            poly = self._chebyshev_features(h, L_hat)
            h    = F.relu(ln(gcn(poly)))
        h = h.mean(dim=1).reshape(B, Tds, -1).mean(dim=1)
        h = self.drop(F.relu(self.ln_fc(self.fc1(h))))
        return self.fc2(h)


class SAMGCN(nn.Module):
    def __init__(self, n_channels, n_classes, T=500, freq_bands=5,
                 gcn_dims=(64,64,64), dropout=0.5, t_stride=8):
        super().__init__()
        self.n_channels = n_channels
        self.freq_bands = freq_bands
        gcn_dim0 = gcn_dims[0]
        self.t_stride = t_stride
        T_pooled = max(T // t_stride, 1)
        k1 = max(T_pooled // 2, 25); k2 = max(T_pooled // 4, 13)
        self.t_conv1 = nn.Conv1d(1, 32, k1, bias=False)
        self.t_conv2 = nn.Conv1d(32, 128, k2, bias=False)
        self.t_mp    = nn.AdaptiveMaxPool1d(1)
        t_out = 128
        self.f_proj  = nn.Sequential(
            nn.Linear(freq_bands, 32), nn.ReLU(), nn.Linear(32, t_out))
        self.cle      = nn.Embedding(n_channels, t_out)
        self.fuse_proj= nn.Linear(t_out*2, gcn_dim0)
        cs_hid = 32
        self.cs_Q  = nn.Linear(gcn_dim0, cs_hid, bias=False)
        self.cs_K  = nn.Linear(gcn_dim0, cs_hid, bias=False)
        self.cs_scale = cs_hid**-0.5
        self.W_alphaT = nn.Parameter(
            torch.eye(n_channels)*0.5 + 0.01*torch.randn(n_channels, n_channels))
        self.W_alphaF = nn.Parameter(
            torch.eye(n_channels)*0.5 + 0.01*torch.randn(n_channels, n_channels))
        self.gcn_T = nn.ModuleList()
        self.gcn_F = nn.ModuleList()
        in_f = gcn_dim0
        for out_f in gcn_dims:
            self.gcn_T.append(nn.Linear(in_f, out_f, bias=False))
            self.gcn_F.append(nn.Linear(in_f, out_f, bias=False))
            in_f = out_f
        total_gcn_feat = sum(gcn_dims)*2*n_channels
        self.drop  = nn.Dropout(dropout)
        self.fc1   = nn.Linear(total_gcn_feat, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2   = nn.Linear(256, n_classes)

    @staticmethod
    def _gcn_step(H, A_raw, W):
        I      = torch.eye(A_raw.shape[-1], device=A_raw.device).unsqueeze(0)
        Atilde = A_raw + I
        D      = Atilde.sum(-1)
        D_is   = torch.diag_embed(1.0 / (D + 1e-8).sqrt())
        A_norm = torch.bmm(torch.bmm(D_is, Atilde), D_is)
        return F.relu(W(torch.bmm(A_norm, H)))

    def _differential_entropy(self, x):
        X_fft = torch.fft.rfft(x, dim=-1)
        n_freq = X_fft.shape[-1]
        edges  = [0, max(int(n_freq*0.05),1), max(int(n_freq*0.10),2),
                  max(int(n_freq*0.17),3), max(int(n_freq*0.40),4), n_freq]
        bands  = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            hi = max(hi, lo+1)
            pwr = (X_fft[:,:,lo:hi].abs()**2).mean(-1)
            bands.append(torch.log(pwr.clamp(min=1e-8)).unsqueeze(-1))
        return torch.cat(bands, dim=-1)

    def forward(self, x):
        B, C, T = x.shape
        device  = x.device
        xc      = x.reshape(B*C, 1, T)
        if self.t_stride > 1:
            xc = F.avg_pool1d(xc, self.t_stride)
        tc = F.relu(self.t_conv1(xc))
        if tc.shape[-1] >= self.t_conv2.kernel_size[0]:
            tc = F.relu(self.t_conv2(tc))
        else:
            tc = F.adaptive_max_pool1d(tc, self.t_conv2.kernel_size[0])
            tc = F.relu(self.t_conv2(tc))
        t_feat = self.t_mp(tc).squeeze(-1).reshape(B, C, 128)
        de     = self._differential_entropy(x)
        f_feat = self.f_proj(de)
        pos    = self.cle(torch.arange(C, device=device))
        t_feat = t_feat + pos.unsqueeze(0)
        H      = self.fuse_proj(torch.cat([t_feat, f_feat], dim=-1))
        Q      = self.cs_Q(H); K_ = self.cs_K(H)
        attn   = torch.softmax(torch.bmm(Q, K_.transpose(1,2))*self.cs_scale, dim=-1)
        Atilde_T = attn * torch.sigmoid(self.W_alphaT).unsqueeze(0)
        Atilde_F = attn * torch.sigmoid(self.W_alphaF).unsqueeze(0)
        H_T, H_F = H, H
        T_outs, F_outs = [], []
        for gcn_t, gcn_f in zip(self.gcn_T, self.gcn_F):
            H_T = self._gcn_step(H_T, Atilde_T, gcn_t)
            H_F = self._gcn_step(H_F, Atilde_F, gcn_f)
            T_outs.append(H_T); F_outs.append(H_F)
        cat  = torch.cat(T_outs + F_outs, dim=-1)
        feat = self.drop(cat.flatten(1))
        feat = F.relu(self.bn_fc(self.fc1(feat)))
        feat = self.drop(feat)
        return self.fc2(feat)


# ===========================================================================
# Dataset
# ===========================================================================

class AugDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.training = True

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment and self.training:
            if np.random.rand() < 0.3:
                x = torch.roll(x, np.random.randint(-5, 6), dims=-1)
            if np.random.rand() < 0.3:
                x = x * np.random.uniform(0.9, 1.1)
            if np.random.rand() < 0.3:
                x = x + torch.randn_like(x) * 0.005
        return x, self.y[idx]


# ===========================================================================
# Data loading  (identical to train_bciciv2a_session_split.py)
# ===========================================================================

def load_subject(subj_id, data_dir):
    """Load T-file only (E-files lack class labels in GDF format).
    Returns (X, y) with per-channel z-score normalisation.
    288 labelled trials per subject (4 classes × 72).
    """
    data_dir = Path(data_dir)

    def _load_raw(path):
        raw = mne.io.read_raw_gdf(str(path), preload=True, verbose=False)
        picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)
        raw.pick(picks)
        raw.filter(8.0, 30.0, fir_design='firwin', verbose=False)
        return raw

    raw_T = _load_raw(data_dir / f"A0{subj_id}T.gdf")

    primary  = {769: 0, 770: 1, 771: 2, 772: 3}
    fallback = {7: 0, 8: 1, 9: 2, 10: 3}

    events, _ = mne.events_from_annotations(raw_T, verbose=False)
    valid = [e for e in events if e[2] in primary]
    if not valid:
        valid = [e for e in events if e[2] in fallback]
        mapping = fallback
    else:
        mapping = primary
    if not valid:
        codes, counts = np.unique(events[:, 2], return_counts=True)
        top4 = codes[np.argsort(counts)[::-1][:4]]
        mapping = {int(c): i for i, c in enumerate(sorted(top4))}
        valid = [e for e in events if e[2] in mapping]

    ev_T = np.array(valid)

    epochs = mne.Epochs(raw_T, ev_T, tmin=0.5, tmax=4.5,
                        baseline=None, preload=True, verbose=False)
    X = epochs.get_data()
    y = np.array([mapping[e[2]] for e in ev_T])

    # Resample to 500 time-points (4 s at 125 Hz) for consistent model input
    from scipy.signal import resample
    X = resample(X, 500, axis=2)

    # Per-channel z-score from T-file statistics
    for ch in range(X.shape[1]):
        mu  = X[:, ch, :].mean()
        std = X[:, ch, :].std()
        if std > 1e-6:
            X[:, ch, :] = (X[:, ch, :] - mu) / std

    return X, y


# ===========================================================================
# LOSO training loop
# ===========================================================================

def train_loso(model_name, data_dir, out_dir, device, epochs=150, patience=25,
               batch_size=32, lr=0.001, dropout=0.5, seed=42):

    np.random.seed(seed); torch.manual_seed(seed)

    subjects  = list(range(1, 10))
    all_X, all_y = {}, {}

    print(f"\nLoading BCI-IV-2a data for all 9 subjects ...")
    for sid in subjects:
        X, y = load_subject(sid, data_dir)
        all_X[sid] = X; all_y[sid] = y
        print(f"  A0{sid}: {len(y)} trials, {X.shape[1]} ch, {X.shape[2]} T")

    n_channels  = next(iter(all_X.values())).shape[1]
    T_len       = next(iter(all_X.values())).shape[2]
    num_classes = 4

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for test_sid in subjects:
        print(f"\n{'='*65}")
        print(f"  LOSO fold: test = A0{test_sid}  |  model = {model_name}")
        print(f"{'='*65}")

        # Build train pool (all other subjects, T+E combined)
        X_tr_list, y_tr_list = [], []
        for sid in subjects:
            if sid == test_sid:
                continue
            X_tr_list.append(all_X[sid])
            y_tr_list.append(all_y[sid])
        X_tr_all = np.concatenate(X_tr_list)
        y_tr_all = np.concatenate(y_tr_list)

        # 20% stratified val split from training set
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        tr_idx, val_idx = next(sss.split(X_tr_all, y_tr_all))
        X_tr,  y_tr  = X_tr_all[tr_idx],  y_tr_all[tr_idx]
        X_val, y_val = X_tr_all[val_idx], y_tr_all[val_idx]

        X_te = all_X[test_sid]
        y_te = all_y[test_sid]

        print(f"  Train: {len(y_tr)}  Val: {len(y_val)}  Test: {len(y_te)}")

        train_ds = AugDataset(X_tr,  y_tr,  augment=True)
        val_ds   = AugDataset(X_val, y_val, augment=False)
        test_ds  = AugDataset(X_te,  y_te,  augment=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

        # Instantiate model
        if model_name == "BrainTopoGCN":
            model = BrainTopoGCN(n_channels, num_classes, T=T_len, dropout=dropout)
        elif model_name == "EEG_GLT-Net":
            model = EEG_GLT_Net(n_channels, num_classes, T=T_len, dropout=dropout)
        elif model_name == "SAMGCN":
            model = SAMGCN(n_channels, num_classes, T=T_len, dropout=dropout)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        model = model.to(device)

        # Class-weighted loss (balanced)
        counts = np.bincount(y_tr, minlength=num_classes).astype(np.float32)
        counts = np.where(counts == 0, 1e6, counts)
        w = torch.tensor(1.0 / (counts + 1e-6)).to(device)
        w = w / w.sum() * num_classes
        criterion = nn.CrossEntropyLoss(weight=w)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5)

        best_val_acc = 0.0
        best_state   = None
        patience_ctr = 0

        for epoch in range(1, epochs + 1):
            model.train(); train_ds.training = True
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval(); train_ds.training = False
            correct = total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = model(xb.to(device)).argmax(1).cpu()
                    correct += (pred == yb).sum().item(); total += len(yb)
            val_acc = 100.0 * correct / total

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = copy.deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1

            if epoch % 25 == 0 or epoch == 1:
                print(f"  ep {epoch:3d}  val={val_acc:.2f}%  best_val={best_val_acc:.2f}%")
            if patience_ctr >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

        # Final test
        model.load_state_dict(best_state)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb.to(device)).argmax(1).cpu()
                correct += (pred == yb).sum().item(); total += len(yb)
        test_acc = 100.0 * correct / total
        print(f"  --> Test acc: {test_acc:.2f}%  (best val: {best_val_acc:.2f}%)")

        r = {"subject": f"A0{test_sid}", "test_acc": test_acc,
             "best_val_acc": best_val_acc}
        results.append(r)

        fold_dir = out_dir / f"fold_A0{test_sid}"
        fold_dir.mkdir(exist_ok=True)
        with open(fold_dir / "result.json", "w") as f:
            json.dump(r, f, indent=2)

    accs     = [r["test_acc"] for r in results]
    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)

    print("\n" + "="*65)
    print(f"RESULTS — {model_name}  BCI-IV-2a LOSO")
    print("="*65)
    for r in results:
        print(f"  {r['subject']}:  {r['test_acc']:.2f}%")
    print(f"  Mean: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print("="*65)
    print(f"\nFor Table 9 (LOSO row):  {model_name}   {mean_acc:.1f} ± {std_acc:.1f}")

    summary = {
        "model":      model_name,
        "protocol":   "LOSO",
        "dataset":    "BCI-IV-2a",
        "n_subjects": len(results),
        "mean_acc":   round(mean_acc, 2),
        "std_acc":    round(std_acc,  2),
        "per_subject": results,
    }
    with open(out_dir / "loso_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved -> {out_dir}/loso_summary.json")
    return summary


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="all",
                        choices=["BrainTopoGCN","EEG_GLT-Net","SAMGCN","all"])
    parser.add_argument("--data_dir",   default="data/bciciv_2a/BCICIV_2a_gdf")
    parser.add_argument("--results_dir",default="results/baselines_bciciv2a_loso")
    parser.add_argument("--epochs",     type=int,   default=150)
    parser.add_argument("--patience",   type=int,   default=25)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--dropout",    type=float, default=0.5)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve data dir relative to script location
    base = Path(__file__).parent.parent
    data_dir = base / args.data_dir
    if not data_dir.exists():
        data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data dir not found: {data_dir}"); return

    models_to_run = (["BrainTopoGCN","EEG_GLT-Net","SAMGCN"]
                     if args.model == "all" else [args.model])

    all_summaries = {}
    for m in models_to_run:
        out_dir = base / args.results_dir / m
        summary = train_loso(
            model_name=m, data_dir=data_dir, out_dir=out_dir,
            device=device, epochs=args.epochs, patience=args.patience,
            batch_size=args.batch_size, lr=args.lr,
            dropout=args.dropout, seed=args.seed)
        all_summaries[m] = summary

    # Final summary table
    if len(all_summaries) > 1:
        print("\n" + "="*65)
        print("ALL BASELINES — BCI-IV-2a LOSO (Table 9 additions)")
        print("="*65)
        for m, s in all_summaries.items():
            print(f"  {m:<20}  {s['mean_acc']:.2f}% ± {s['std_acc']:.2f}%")
        print("="*65)


if __name__ == "__main__":
    main()
