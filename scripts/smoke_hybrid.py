"""
Quick 2-subject smoke test for HyperLorentzNet-HGCN.
Runs NIS007 and NIS008 (historically best subjects) with the fixed model
(class-weighted loss + sign-only PLV gating + 2 GNN layers).

Usage:
    python scripts/smoke_hybrid.py --data_dir data/steele
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.lorentz_tcnet.graph import TriLayerGraphBuilder
from src.lorentz_tcnet.model_hybrid import HyperLorentzNetHGCN
from src.lorentz_tcnet.pathway import PathwayExtractor
from torch.utils.data import ConcatDataset
from scripts.train_hybrid_loso import AugDataset

SUBJECTS = [f"NIS{i:03d}" for i in range(1, 11)]
SMOKE_SUBJECTS = ["NIS007", "NIS008"]   # historically strongest
N_EEG, N_ESG, N_EMG = 28, 15, 8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/steele")
    p.add_argument("--results_dir",default="results/smoke_hybrid")
    p.add_argument("--epochs",     default=40,   type=int)
    p.add_argument("--patience",   default=12,   type=int)
    p.add_argument("--batch_size", default=64,   type=int)
    p.add_argument("--lr",         default=3e-4, type=float)
    p.add_argument("--hidden_dim", default=64,   type=int)
    p.add_argument("--latent_dim", default=48,   type=int)
    p.add_argument("--gnn_hidden", default=48,   type=int)
    p.add_argument("--gnn_layers", default=2,    type=int)
    p.add_argument("--gnn_heads",  default=2,    type=int)
    p.add_argument("--k_cross",    default=6,    type=int)
    p.add_argument("--dropout",    default=0.3,  type=float)
    p.add_argument("--t_stride",   default=4,    type=int,
                   help="Temporal downsampling: T=1000 -> T//t_stride (default=4, gives ~4x speedup)")
    p.add_argument("--seed",       default=42,   type=int)
    return p.parse_args()


def load_subject(data_dir, subj):
    d = np.load(os.path.join(data_dir, f"{subj}.npz"), allow_pickle=True)
    return d["eeg"], d["esg"], d["emg"], d["labels"]


def run_fold(test_subj, args, device):
    train_subjs = [s for s in SUBJECTS if s != test_subj]
    # Load per-subject arrays; keep them separate to avoid 1.8 GB concatenation
    parts = [load_subject(args.data_dir, s) for s in train_subjs]
    # Labels only — tiny int arrays, safe to concatenate
    lab_tr = np.concatenate([p[3] for p in parts])
    eeg_te, esg_te, emg_te, lab_te = load_subject(args.data_dir, test_subj)

    num_classes = len(np.unique(lab_tr))
    n_train = len(lab_tr)
    print(f"\n{'='*55}\nFOLD: {test_subj}  |  train={n_train}  test={len(lab_te)}  cls={num_classes}\n{'='*55}")

    # ── PLV graph — sample ~30 trials per subject (avoids OOM) ──
    t0 = time.time()
    builder = TriLayerGraphBuilder(N_EEG, N_ESG, N_EMG)
    n_per = max(2, 256 // len(parts))
    eeg_g = np.concatenate([p[0][:n_per] for p in parts])
    esg_g = np.concatenate([p[1][:n_per] for p in parts])
    emg_g = np.concatenate([p[2][:n_per] for p in parts])
    edge_index, edge_weight = builder.build(eeg_g, esg_g, emg_g, k_cross=args.k_cross, signed=True)
    del eeg_g, esg_g, emg_g
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    print(f"  PLV graph: {edge_index.shape[1]} edges  ({time.time()-t0:.1f}s)")

    # ── Model ──
    model = HyperLorentzNetHGCN(
        eeg_channels=N_EEG, esg_channels=N_ESG, emg_channels=N_EMG,
        hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
        num_classes=num_classes,
        gnn_hidden=args.gnn_hidden, gnn_layers=args.gnn_layers,
        gnn_heads=args.gnn_heads, dropout=args.dropout,
        use_stage1_logits=True, t_stride=args.t_stride,
    ).to(device)
    model.register_graph(edge_index, edge_weight)
    T_eff = 1000 // args.t_stride
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}  | T_eff={T_eff} (stride={args.t_stride})")

    # ── Class-weighted loss ──
    counts = np.bincount(lab_tr, minlength=num_classes).astype(np.float32)
    w = torch.tensor(1.0 / (counts + 1e-6)).to(device)
    w = w / w.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=w)
    print(f"  Class weights: {[round(x,3) for x in w.cpu().tolist()]}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    train_loader = DataLoader(
        ConcatDataset([AugDataset(p[0], p[1], p[2], p[3], augment=True)  for p in parts]),
        batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(
        AugDataset(eeg_te, esg_te, emg_te, lab_te, augment=False),
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    best_acc, best_state, patience_ctr = 0.0, None, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for eeg_b, esg_b, emg_b, lab_b in train_loader:
            eeg_b, esg_b, emg_b, lab_b = eeg_b.to(device), esg_b.to(device), emg_b.to(device), lab_b.to(device)
            optimizer.zero_grad()
            out = model(eeg_b, esg_b, emg_b)
            loss = criterion(out["logits"], lab_b) + 0.2 * criterion(out["stage1_logits"], lab_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = tot = 0
        per_class_correct = np.zeros(num_classes)
        per_class_total   = np.zeros(num_classes)
        with torch.no_grad():
            for eeg_b, esg_b, emg_b, lab_b in test_loader:
                eeg_b, esg_b, emg_b, lab_b = eeg_b.to(device), esg_b.to(device), emg_b.to(device), lab_b.to(device)
                preds = model(eeg_b, esg_b, emg_b)["logits"].argmax(-1)
                correct += (preds == lab_b).sum().item()
                tot     += lab_b.shape[0]
                for c in range(num_classes):
                    mask = (lab_b == c)
                    per_class_correct[c] += (preds[mask] == lab_b[mask]).sum().item()
                    per_class_total[c]   += mask.sum().item()
        acc = 100.0 * correct / tot
        per_cls_acc = [round(float(100 * per_class_correct[c] / max(float(per_class_total[c]), 1.0)), 1) for c in range(num_classes)]
        history.append({"epoch": epoch, "loss": round(total_loss/len(train_loader),4), "acc": round(acc,2)})

        improved = acc > best_acc
        if epoch % 10 == 0 or improved:
            tag = "  <--best" if improved else ""
            print(f"  Ep {epoch:3d}  loss={total_loss/len(train_loader):.4f}  acc={acc:.2f}%  cls={per_cls_acc}{tag}")
        if improved:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  Early stop @ epoch {epoch}")
                break

    # ── Reload best, extract pathways ──
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    extractor = PathwayExtractor(model, edge_index.cpu(), n_eeg=N_EEG, n_esg=N_ESG, n_emg=N_EMG)
    with torch.no_grad():
        for eeg_b, esg_b, emg_b, lab_b in test_loader:
            extractor.accumulate((eeg_b, esg_b, emg_b, lab_b), device=str(device))

    fold_dir = os.path.join(args.results_dir, f"fold_{test_subj}")
    os.makedirs(fold_dir, exist_ok=True)

    print(f"\n  >> {test_subj} top pathways per class:")
    all_pathways = {}
    for cls in range(num_classes):
        try:
            pw = extractor.top_pathways(class_idx=cls, top_k=5)
            all_pathways[cls] = pw
            top = pw[0] if pw else None
            if top:
                print(f"    Class {cls}: {top['eeg_label']}->{top['esg_label']}->{top['emg_label']}  chain={top['chain_score']:.5f}  attn={top['attn_eeg_esg']:.4f}/{top['attn_esg_emg']:.4f}")
            extractor.export_csv(pw, os.path.join(fold_dir, f"pathways_class{cls}.csv"))
        except ValueError as e:
            print(f"    Class {cls}: {e}")

    print(f"\n  [OK] {test_subj}  BEST ACC = {best_acc:.2f}%\n")
    return {"subject": test_subj, "best_acc": best_acc, "history": history, "pathways": all_pathways}


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    os.makedirs(args.results_dir, exist_ok=True)
    results = []
    for subj in SMOKE_SUBJECTS:
        r = run_fold(subj, args, device)
        results.append(r)

    accs = [r["best_acc"] for r in results]
    print(f"\n{'='*55}")
    print(f"SMOKE TEST SUMMARY")
    print(f"{'='*55}")
    for r in results:
        print(f"  {r['subject']}: {r['best_acc']:.2f}%")
    print(f"  Mean: {np.mean(accs):.2f}%  |  Std: {np.std(accs):.2f}%")
    print(f"{'='*55}")

    with open(os.path.join(args.results_dir, "smoke_summary.json"), "w") as f:
        json.dump({"subjects": SMOKE_SUBJECTS, "accs": accs,
                   "mean": float(np.mean(accs)), "std": float(np.std(accs)),
                   "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
