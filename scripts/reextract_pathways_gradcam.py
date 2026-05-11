"""
Re-extract class-discriminative pathways using Node-GradCAM.

WHY THIS APPROACH
-----------------
The v3 model's class-prototype conditioning (cond_proj) was zero-initialised
and learned only small residual biases, so the returned attn_weights do not
differ across movement classes (Jaccard=1.0 in the original extraction).

Node-GradCAM is the correct fix:
  - Forward the batch with grad tracking enabled
  - Compute: importance[node,class] = mean_batch(relu(d logit_c / d node_feats
                                                        * node_feats))
  - For each edge (i,j): edge_imp_c = sqrt(imp_i_c * imp_j_c )
  - Class-discriminative by construction — no re-training needed.

This is also the more principled approach for publication (standard GradCAM
attribution, widely adopted in BCI interpretability literature).

OUTPUT PER FOLD
---------------
  pathways_class{c}.csv              — top-k EEG->ESG->EMG chain pathways
  node_importance_class{c}.npy       — (N=51,) raw node importance array
  pathway_discriminability.json      — mean |diff| and Jaccard per fold

Usage
-----
    # All 10 folds (recommended — runs in ~5 min)
    python scripts/reextract_pathways_gradcam.py

    # Single fold for quick validation
    python scripts/reextract_pathways_gradcam.py --subjects NIS008

    # Different results dir
    python scripts/reextract_pathways_gradcam.py --results_dir results/hybrid_loso_v4
"""
from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lorentz_tcnet.model_hybrid import HyperLorentzNetHGCN

# ---------------------------------------------------------------------------
SUBJECTS   = [f"NIS{i:03d}" for i in range(1, 11)]
N_EEG, N_ESG, N_EMG = 28, 15, 8


def parse_args():
    p = argparse.ArgumentParser(description="Node-GradCAM pathway re-extraction")
    p.add_argument("--results_dir", default="results/hybrid_loso_v3")
    p.add_argument("--data_dir",    default="data/steele")
    p.add_argument("--subjects",    nargs="+", default=None)
    # Must match checkpoint hyperparams (v3 defaults)
    p.add_argument("--hidden_dim",  default=64,  type=int)
    p.add_argument("--latent_dim",  default=48,  type=int)
    p.add_argument("--gnn_hidden",  default=48,  type=int)
    p.add_argument("--gnn_layers",  default=1,   type=int)
    p.add_argument("--gnn_heads",   default=2,   type=int)
    p.add_argument("--dropout",     default=0.5, type=float)
    p.add_argument("--t_stride",    default=4,   type=int)
    p.add_argument("--batch_size",  default=16,  type=int,
                   help="Smaller batch is fine; GradCAM is memory-bounded")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--top_k",       default=10,  type=int)
    p.add_argument("--jaccard_k",   default=5,   type=int)
    p.add_argument("--train_n",     default=30,  type=int,
                   help="Training samples per subject to supplement test set")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_subject(data_dir, subj):
    d = np.load(os.path.join(data_dir, f"{subj}.npz"), allow_pickle=True)
    return d["eeg"], d["esg"], d["emg"], d["labels"]


def infer_num_classes(state_dict: dict) -> int:
    key = "stage2.classifier.2.weight"
    if key in state_dict:
        return int(state_dict[key].shape[0])
    # Fallback: scan for a small-output bias vector
    for k, v in state_dict.items():
        if k.endswith(".bias") and v.ndim == 1 and 2 <= int(v.shape[0]) <= 8:
            return int(v.shape[0])
    raise RuntimeError("Cannot infer num_classes from checkpoint state dict.")


def jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def top_k_eeg_nodes(ni: np.ndarray, n_eeg: int, k: int) -> set:
    return set(int(x) for x in np.argsort(ni[:n_eeg])[-k:])


# ---------------------------------------------------------------------------
# Node-GradCAM extractor
# ---------------------------------------------------------------------------

class NodeGradCamExtractor:
    """
    Input-GradCAM: computes d(logit_c)/d(raw_input_signals) so importance
    is non-zero regardless of which stage (Stage1 or Stage2) dominates.

    For each class c and each sample of that class:
        grad_eeg = d(logit_c) / d(eeg)    (B, C_eeg, T)
        imp_eeg_ch = relu(grad_eeg * eeg).mean(time)   -> (C_eeg,) per sample

    Same for ESG and EMG.  Concatenate -> (N,) node importance.
    Edge importance = sqrt(imp_src * imp_dst).

    This captures Stage-1-dominant models (NIS008 78%) AND Stage-2-dominant
    models equally well, since the raw input is always a leaf in the comp graph.
    """

    def __init__(self, model, edge_index: torch.Tensor,
                 n_eeg=28, n_esg=15, n_emg=8):
        self.model      = model
        self.edge_index = edge_index       # (2, E) on CPU
        self.n_eeg      = n_eeg
        self.n_esg      = n_esg
        self.n_emg      = n_emg
        self.N          = n_eeg + n_esg + n_emg

        self._imp_sum   = {}   # cls -> np.ndarray (N,)
        self._imp_count = {}   # cls -> int

    @torch.enable_grad()
    def accumulate(self, eeg, esg, emg, labels, device):
        """
        Process one batch (CPU tensors), accumulate Input-GradCAM per class.
        """
        self.model.eval()
        self.model.to(device)

        labs_np  = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
        labs_dev = torch.as_tensor(labs_np, dtype=torch.long, device=device)

        for cls in sorted(np.unique(labs_np).tolist()):
            mask  = labs_dev == cls
            n_cls = int(mask.sum().item())
            if n_cls == 0:
                continue

            # Move to device with grad enabled — inputs are leaves
            eeg_c = eeg[mask.cpu()].to(device).requires_grad_(True)
            esg_c = esg[mask.cpu()].to(device).requires_grad_(True) if esg is not None else None
            emg_c = emg[mask.cpu()].to(device).requires_grad_(True) if emg is not None else None

            self.model.zero_grad()
            out = self.model(eeg_c, esg_c, emg_c)

            # Sum logit_cls over all samples of cls in this batch (scalar)
            class_score = out["logits"][:, cls].sum()
            class_score.backward()

            # Per-input GradCAM: relu(grad * input).mean(time) -> channel importance
            def _chan_imp(x_in, grad):
                if x_in is None or grad is None:
                    return None
                # x_in, grad: (n_cls, C, T) -> (C,)
                return torch.relu(grad.detach() * x_in.detach()).mean(dim=-1).mean(dim=0)

            imp_eeg = _chan_imp(eeg_c, eeg_c.grad)   # (n_eeg,)
            imp_esg = _chan_imp(esg_c, esg_c.grad) if esg_c is not None else torch.zeros(self.n_esg)
            imp_emg = _chan_imp(emg_c, emg_c.grad) if emg_c is not None else torch.zeros(self.n_emg)

            if imp_eeg is None:
                print(f"  [Warning] eeg.grad is None for class {cls} — skipping")
                continue

            # Concatenate to (N,) node importance
            imp_node = torch.cat([imp_eeg.cpu(),
                                   imp_esg.cpu() if imp_esg is not None else torch.zeros(self.n_esg),
                                   imp_emg.cpu() if imp_emg is not None else torch.zeros(self.n_emg)
                                   ]).numpy()  # (N,)

            if cls not in self._imp_sum:
                self._imp_sum[cls]   = np.zeros(self.N, dtype=np.float64)
                self._imp_count[cls] = 0
            self._imp_sum[cls]   += imp_node.astype(np.float64) * n_cls
            self._imp_count[cls] += n_cls

    def avg_node_importance(self, cls: int) -> np.ndarray:
        """(N,) average node importance for the given class."""
        cnt = self._imp_count.get(cls, 0)
        if cnt == 0:
            raise ValueError(f"No samples for class {cls}")
        return (self._imp_sum[cls] / cnt).astype(np.float32)

    def edge_importance(self, cls: int) -> np.ndarray:
        """(E,) geometric-mean edge importance derived from node importances."""
        ni   = self.avg_node_importance(cls).astype(np.float64)
        src  = self.edge_index[0].numpy()
        dst  = self.edge_index[1].numpy()
        return np.sqrt(np.maximum(ni[src], 0.0) * np.maximum(ni[dst], 0.0)).astype(np.float32)

    # ---- Pathway extraction --------------------------------------------

    def top_pathways(self, cls: int, top_k: int = 10,
                     min_imp: float = 0.0) -> list[dict]:
        """Chain EEG->ESG->EMG by geometric-mean edge importance."""
        ei   = self.edge_importance(cls)
        src  = self.edge_index[0].numpy()
        dst  = self.edge_index[1].numpy()

        off_esg = self.n_eeg
        off_emg = self.n_eeg + self.n_esg

        # Edges: EEG -> ESG
        ee_esg = [
            (src[e], dst[e] - off_esg, float(ei[e]))
            for e in range(len(src))
            if src[e] < self.n_eeg and off_esg <= dst[e] < off_emg
            and ei[e] >= min_imp
        ]
        # Edges: ESG -> EMG  (lookup by esg node)
        esg_to_emg: dict[int, list[tuple[int, float]]] = {}
        for e in range(len(src)):
            if off_esg <= src[e] < off_emg and dst[e] >= off_emg and ei[e] >= min_imp:
                esg_node = src[e] - off_esg
                emg_node = dst[e] - off_emg
                esg_to_emg.setdefault(esg_node, []).append((emg_node, float(ei[e])))

        chains = []
        for (eeg_n, esg_n, a1) in ee_esg:
            if esg_n not in esg_to_emg:
                continue
            best_emg, a2 = max(esg_to_emg[esg_n], key=lambda x: x[1])
            chains.append({
                "eeg_node":   int(eeg_n),
                "esg_node":   int(esg_n),
                "emg_node":   int(best_emg),
                "eeg_label":  f"EEG{eeg_n}",
                "esg_label":  f"ESG{esg_n}",
                "emg_label":  f"EMG{best_emg}",
                "imp_eeg_esg": float(a1),
                "imp_esg_emg": float(a2),
                "chain_score": float(a1 * a2),
            })

        chains.sort(key=lambda x: x["chain_score"], reverse=True)
        return chains[:top_k]

    def print_pathways(self, pathways: list[dict], class_name: str = "") -> None:
        print(f"\n=== Top Pathways {class_name} ===")
        print(f"{'Rank':<5} {'EEG':<8} {'ESG':<8} {'EMG':<8} "
              f"{'imp_E-E':>10} {'imp_E-M':>10} {'chain':>12}")
        print("-" * 65)
        for i, pw in enumerate(pathways, 1):
            print(f"{i:<5} {pw['eeg_label']:<8} {pw['esg_label']:<8} "
                  f"{pw['emg_label']:<8} "
                  f"{pw['imp_eeg_esg']:>10.5f} {pw['imp_esg_emg']:>10.5f} "
                  f"{pw['chain_score']:>12.7f}")

    def export_csv(self, pathways: list[dict], path: str) -> None:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="") as f:
            if not pathways:
                f.write("# No pathways found\n")
                return
            w = csv.DictWriter(f, fieldnames=list(pathways[0].keys()))
            w.writeheader()
            w.writerows(pathways)
        print(f"  Saved {len(pathways)} pathways -> {path}")


# ---------------------------------------------------------------------------
# Per-fold extraction
# ---------------------------------------------------------------------------

def reextract_fold(subj: str, args, device: torch.device) -> dict:
    fold_dir  = os.path.join(args.results_dir, f"fold_{subj}")
    ckpt_path = os.path.join(fold_dir, "best_model.pt")

    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] {ckpt_path} not found")
        return {}

    print(f"\n{'='*60}")
    print(f"  Node-GradCAM  |  {subj}  |  {fold_dir}")
    print(f"{'='*60}")

    ckpt        = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state       = ckpt["model_state"]
    ei          = ckpt["edge_index"]    # (2, E) CPU
    ew          = ckpt["edge_weight"]   # (E,)   CPU
    num_classes = infer_num_classes(state)
    print(f"  num_classes={num_classes}  edges={ei.shape[1]}")

    model = HyperLorentzNetHGCN(
        eeg_channels=N_EEG, esg_channels=N_ESG, emg_channels=N_EMG,
        hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
        num_classes=num_classes,    gnn_hidden=args.gnn_hidden,
        gnn_layers=args.gnn_layers, gnn_heads=args.gnn_heads,
        dropout=args.dropout, use_stage1_logits=True, t_stride=args.t_stride,
    )
    model.register_graph(ei.to(device), ew.to(device))
    model.load_state_dict(state)
    model = model.to(device)
    print(f"  Loaded  ({sum(p.numel() for p in model.parameters()):,} params)")

    # ---- Build data pool ----
    eeg_te, esg_te, emg_te, lab_te = load_subject(args.data_dir, subj)
    train_extras = [load_subject(args.data_dir, s)
                    for s in SUBJECTS if s != subj]
    n = args.train_n
    eeg_all = np.concatenate([eeg_te] + [e[0][:n] for e in train_extras])
    esg_all = np.concatenate([esg_te] + [e[1][:n] for e in train_extras])
    emg_all = np.concatenate([emg_te] + [e[2][:n] for e in train_extras])
    lab_all = np.concatenate([lab_te] + [e[3][:n] for e in train_extras])
    print(f"  Pool: {len(lab_all)} samples  |  classes: {sorted(np.unique(lab_all).tolist())}")

    extractor = NodeGradCamExtractor(
        model=model, edge_index=ei,
        n_eeg=N_EEG, n_esg=N_ESG, n_emg=N_EMG
    )

    bs    = args.batch_size
    eeg_t = torch.from_numpy(eeg_all.astype(np.float32))
    esg_t = torch.from_numpy(esg_all.astype(np.float32))
    emg_t = torch.from_numpy(emg_all.astype(np.float32))
    lab_t = torch.from_numpy(lab_all.astype(np.int64))

    print("  Accumulating Node-GradCAM ...")
    for start in range(0, len(lab_all), bs):
        end = min(start + bs, len(lab_all))
        extractor.accumulate(
            eeg_t[start:end], esg_t[start:end],
            emg_t[start:end], lab_t[start:end],
            str(device)
        )

    # ---- Discriminability metrics ----
    present = sorted(extractor._imp_count.keys())
    print(f"  Classes with gradients: {present}")

    if len(present) < 2:
        print("  [Warning] Need >= 2 classes for discriminability check")
        return {}

    ni = {c: extractor.avg_node_importance(c) for c in present}
    ei_imp = {c: extractor.edge_importance(c) for c in present}

    pair_diffs = []
    for ci, cj in combinations(present, 2):
        d = float(np.abs(ei_imp[ci] - ei_imp[cj]).mean())
        pair_diffs.append((ci, cj, d))

    jaccards = []
    for ci, cj in combinations(present, 2):
        si = top_k_eeg_nodes(ni[ci], N_EEG, args.jaccard_k)
        sj = top_k_eeg_nodes(ni[cj], N_EEG, args.jaccard_k)
        jaccards.append(jaccard(si, sj))

    mean_diff = float(np.mean([d for *_, d in pair_diffs]))
    mean_jacc = float(np.mean(jaccards))

    print(f"\n  [DISCRIMINABILITY (NodeGradCAM)]")
    for ci, cj, d in pair_diffs:
        print(f"    |imp_{ci} - imp_{cj}| = {d:.6f}")
    for (ci, cj), j in zip(combinations(present, 2), jaccards):
        print(f"    Jaccard top-{args.jaccard_k} EEG (cls {ci} vs {cj}) = {j:.3f}")
    print(f"  Mean |diff| = {mean_diff:.6f}  |  Mean Jaccard = {mean_jacc:.3f}")
    print(f"  (Old attn_weights: diff=0.000, Jaccard=1.000)")

    # ---- Save pathways + node importance arrays ----
    for cls in present:
        try:
            pw = extractor.top_pathways(cls, top_k=args.top_k)
            extractor.print_pathways(pw, class_name=f"Class {cls}")
            extractor.export_csv(pw, os.path.join(fold_dir, f"pathways_class{cls}.csv"))
            np.save(os.path.join(fold_dir, f"node_importance_class{cls}.npy"), ni[cls])
        except Exception as e:
            print(f"  [Warning cls {cls}] {e}")

    metrics = {
        "subject":          subj,
        "method":           "NodeGradCAM",
        "mean_edge_diff":   mean_diff,
        "mean_jaccard":     mean_jacc,
        "pair_diffs":       [(int(a), int(b), d) for a, b, d in pair_diffs],
        "pair_jaccards":    [(int(a), int(b), float(j))
                             for (a, b), j in zip(combinations(present, 2), jaccards)],
    }
    with open(os.path.join(fold_dir, "pathway_discriminability.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    res_path = os.path.join(fold_dir, "fold_result.json")
    if os.path.exists(res_path):
        with open(res_path) as f:
            res = json.load(f)
        res["attn_per_class"]          = {str(c): ei_imp[c].tolist() for c in present}
        res["pathway_discriminability"] = metrics
        with open(res_path, "w") as f:
            json.dump(res, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
def main():
    args     = parse_args()
    device   = torch.device(args.device if torch.cuda.is_available() else "cpu")
    subjects = args.subjects or SUBJECTS
    print(f"Device={device}  results_dir={args.results_dir}")
    print(f"Subjects: {subjects}")

    summary = []
    for subj in subjects:
        m = reextract_fold(subj, args, device)
        if m:
            summary.append(m)

    if not summary:
        print("Nothing processed.")
        return

    print(f"\n{'='*70}")
    print("  NODE-GRADCAM DISCRIMINABILITY SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Subject':<10} {'MeanEdgeDiff':>14} {'MeanJaccard':>13}  Status")
    print(f"  {'-'*10} {'-'*14} {'-'*13}  ------")
    for m in summary:
        d  = m["mean_edge_diff"]
        j  = m["mean_jaccard"]
        ok = "OK" if d > 1e-5 and j < 0.85 else "POOR"
        print(f"  {m['subject']:<10} {d:>14.6f} {j:>13.3f}  {ok}")

    print(f"\n  Grand mean:")
    print(f"    |edge diff|  = {np.mean([m['mean_edge_diff'] for m in summary]):.6f}")
    print(f"    Jaccard      = {np.mean([m['mean_jaccard'] for m in summary]):.3f}")
    print("\nDone. pathways_class*.csv + node_importance_class*.npy written.")


if __name__ == "__main__":
    main()
