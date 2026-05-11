"""
Pathway Extraction Utilities for HyperLorentzNet-HGCN.

Extracts "brain → spine → muscle" functional pathways from trained attention
weights. For each movement class you get a ranked list of:

    EEG-ch_i  →  ESG-ch_j  →  EMG-ch_k   (chain_score)

These represent the dominant functional connective routes for that class.

Usage
-----
    from src.lorentz_tcnet.pathway import PathwayExtractor

    extractor = PathwayExtractor(
        model=trained_hybrid_model,
        edge_index=edge_index,         # (2, E)
        n_eeg=28, n_esg=15, n_emg=8,
        # Optional channel labels
        eeg_labels=[f"EEG-{i}" for i in range(28)],
        esg_labels=[f"ESG-{i}" for i in range(15)],
        emg_labels=[f"EMG-{i}" for i in range(8)],
    )

    # Run inference on test data and accumulate attention per class
    extractor.accumulate(data_loader, device='cuda')

    # Get top pathways for class 2 (ankle dorsiflexion, etc.)
    pathways = extractor.top_pathways(class_idx=2, top_k=10)
    extractor.print_pathways(pathways)

    # Export to CSV for paper Table C
    extractor.export_csv(pathways, path="pathway_class2.csv")

    # Compute edge–edge stability across LOSO folds
    extractor.fold_stability(all_fold_edge_attns)
"""
from __future__ import annotations

import csv
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class PathwayExtractor:
    """
    Accumulates class-conditioned attention weights over a dataset and
    extracts ranked EEG → ESG → EMG pathway chains.
    """

    def __init__(
        self,
        model: nn.Module,
        edge_index: torch.Tensor,   # (2, E)
        n_eeg: int,
        n_esg: int,
        n_emg: int,
        eeg_labels: Optional[List[str]] = None,
        esg_labels: Optional[List[str]] = None,
        emg_labels: Optional[List[str]] = None,
    ):
        self.model = model
        self.edge_index = edge_index       # (2, E)
        self.n_eeg = n_eeg
        self.n_esg = n_esg
        self.n_emg = n_emg
        self.E = edge_index.shape[1]

        # Offsets
        self._off_esg = n_eeg
        self._off_emg = n_eeg + n_esg

        # Labels
        self.eeg_labels = eeg_labels or [f"EEG{i}" for i in range(n_eeg)]
        self.esg_labels = esg_labels or [f"ESG{i}" for i in range(n_esg)]
        self.emg_labels = emg_labels or [f"EMG{i}" for i in range(n_emg)]

        # Accumulators: class → sum of attention vectors, count
        self._attn_sum: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(self.E))
        self._attn_count: Dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def accumulate(
        self,
        data: Tuple,                     # (eeg, esg, emg, labels) tensors
        device: str = "cpu",
    ) -> None:
        """
        Run inference on a batch or full dataset and accumulate per-class attention.

        Parameters
        ----------
        data : tuple of (eeg, esg, emg, labels)
               – tensors already loaded; use accumulate_loader for DataLoader
        device : torch device string
        """
        eeg, esg, emg, labels = data
        eeg = eeg.to(device)
        esg = esg.to(device) if esg is not None else None
        emg = emg.to(device) if emg is not None else None

        self.model.eval()
        self.model.to(device)

        # Pass true labels for hard one-hot prototype conditioning so that
        # each sample's GNN attention is biased toward its true movement class.
        # This is the key step that makes pathways class-discriminative.
        lab_device = (labels.to(device).long()
                      if isinstance(labels, torch.Tensor)
                      else torch.tensor(labels, dtype=torch.long, device=device))
        out = self.model(eeg, esg, emg, true_label=lab_device)
        attn = out.get("attn_weights")

        if attn is None:
            raise RuntimeError("Model returned no attention weights. "
                               "Make sure graph is registered and Stage 2 is active.")

        # attn: (B, E) per-sample attention weights
        attn_np = attn.cpu().numpy()               # (B, E)
        if attn_np.ndim == 1:                      # backwards-compat guard
            attn_np = attn_np[np.newaxis, :].repeat(len(labels), axis=0)

        # Use TRUE labels for class-conditional pathway discovery
        labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)

        # Accumulate per TRUE class (use ground truth, not predictions)
        for b in range(len(labels_np)):
            cls = int(labels_np[b])
            self._attn_sum[cls] += attn_np[b]      # (E,) for this sample
            self._attn_count[cls] += 1

    @torch.no_grad()
    def accumulate_loader(
        self,
        loader: DataLoader,
        device: str = "cpu",
    ) -> None:
        """Iterate over a DataLoader and accumulate attention per class."""
        self.model.eval()
        self.model.to(device)
        for batch in loader:
            if len(batch) == 4:
                eeg, esg, emg, labels = batch
            elif len(batch) == 3:
                eeg, labels = batch[0], batch[-1]
                esg = emg = None
            else:
                raise ValueError("Expected 3 or 4 tensors in batch")
            self.accumulate((eeg, esg, emg, labels), device=device)

    def avg_attention(self, class_idx: int) -> np.ndarray:
        """Return average attention vector for a given class. Shape: (E,)"""
        cnt = self._attn_count[class_idx]
        if cnt == 0:
            raise ValueError(f"No samples accumulated for class {class_idx}")
        return self._attn_sum[class_idx] / cnt

    # ------------------------------------------------------------------
    # Pathway extraction
    # ------------------------------------------------------------------

    def top_pathways(
        self,
        class_idx: int,
        top_k: int = 10,
        min_edge_attn: float = 0.01,
    ) -> List[Dict]:
        """
        Chain EEG → ESG → EMG top-attention edges into ranked pathways.

        Algorithm
        ---------
        1. Get averaged attention per edge for this class.
        2. Separate edges into EEG→ESG, ESG→EMG, EEG→EMG groups.
        3. For each EEG→ESG edge (i,j) with attention a1:
             look for best ESG→EMG edge (j,k) with attention a2.
             chain_score = a1 * a2.
        4. Rank by chain_score.

        Parameters
        ----------
        class_idx     : movement / class label to analyse
        top_k         : number of pathways to return
        min_edge_attn : skip edges weaker than this threshold

        Returns
        -------
        list of dicts:
            eeg_node, esg_node, emg_node  (global indices)
            eeg_label, esg_label, emg_label (human-readable)
            attn_eeg_esg, attn_esg_emg, chain_score
        """
        attn = self.avg_attention(class_idx)       # (E,)
        src = self.edge_index[0].numpy()
        dst = self.edge_index[1].numpy()

        # Classify each edge
        # EEG→ESG: src in EEG range, dst in ESG range
        eeg_esg_mask = (
            (src < self.n_eeg) & (dst >= self._off_esg) & (dst < self._off_emg)
        ) | (
            (dst < self.n_eeg) & (src >= self._off_esg) & (src < self._off_emg)
        )
        esg_emg_mask = (
            (src >= self._off_esg) & (src < self._off_emg) & (dst >= self._off_emg)
        ) | (
            (dst >= self._off_esg) & (dst < self._off_emg) & (src >= self._off_emg)
        )

        # Build ESG→EMG lookup: esg_local_idx → (emg_local_idx, attn_val)
        esg_to_emg: Dict[int, Tuple[int, float]] = {}
        for e in np.where(esg_emg_mask)[0]:
            a = float(attn[e])
            if a < min_edge_attn:
                continue
            s, d = int(src[e]), int(dst[e])
            # Normalise to (esg_local, emg_local)
            if s >= self._off_esg and d >= self._off_emg:
                esg_l, emg_l = s - self._off_esg, d - self._off_emg
            else:
                esg_l, emg_l = d - self._off_esg, s - self._off_emg
            if esg_l not in esg_to_emg or a > esg_to_emg[esg_l][1]:
                esg_to_emg[esg_l] = (emg_l, a)

        pathways = []
        for e in np.where(eeg_esg_mask)[0]:
            a1 = float(attn[e])
            if a1 < min_edge_attn:
                continue
            s, d = int(src[e]), int(dst[e])
            if s < self.n_eeg:
                eeg_l, esg_l = s, d - self._off_esg
            else:
                eeg_l, esg_l = d, s - self._off_esg

            if esg_l in esg_to_emg:
                emg_l, a2 = esg_to_emg[esg_l]
                chain_score = a1 * a2
                pathways.append({
                    "eeg_node": eeg_l,
                    "esg_node": esg_l,
                    "emg_node": emg_l,
                    "eeg_label": self.eeg_labels[eeg_l],
                    "esg_label": self.esg_labels[esg_l],
                    "emg_label": self.emg_labels[emg_l],
                    "attn_eeg_esg": round(a1, 5),
                    "attn_esg_emg": round(a2, 5),
                    "chain_score": round(chain_score, 7),
                })

        pathways.sort(key=lambda x: x["chain_score"], reverse=True)
        return pathways[:top_k]

    # ------------------------------------------------------------------
    # Stability analysis (across LOSO folds)
    # ------------------------------------------------------------------

    @staticmethod
    def fold_stability(
        fold_attns: List[np.ndarray],
        top_k: int = 20,
    ) -> Dict[str, float]:
        """
        Compute Jaccard similarity of top-k edges across folds.

        Parameters
        ----------
        fold_attns : list of (E,) arrays — one per fold (or per class per fold)
        top_k      : how many top edges to use for Jaccard

        Returns
        -------
        dict with 'mean_jaccard', 'std_jaccard' across fold pairs
        """
        n = len(fold_attns)
        top_sets = []
        for a in fold_attns:
            top_idx = set(np.argsort(a)[::-1][:top_k].tolist())
            top_sets.append(top_idx)

        jaccards = []
        for i in range(n):
            for j in range(i + 1, n):
                inter = len(top_sets[i] & top_sets[j])
                union = len(top_sets[i] | top_sets[j])
                jaccards.append(inter / union if union > 0 else 0.0)

        return {
            "mean_jaccard": float(np.mean(jaccards)),
            "std_jaccard": float(np.std(jaccards)),
            "n_folds": n,
        }

    # ------------------------------------------------------------------
    # Printing / export
    # ------------------------------------------------------------------

    def print_pathways(self, pathways: List[Dict], class_name: str = "") -> None:
        """Pretty-print pathway table."""
        header = f"{'Rank':<5} {'EEG':<10} {'ESG':<10} {'EMG':<10} {'attn_E-E':<12} {'attn_E-M':<12} {'chain':<12}"
        print(f"\n=== Top Pathways {class_name} ===")
        print(header)
        print("-" * len(header))
        for rank, p in enumerate(pathways, 1):
            print(
                f"{rank:<5} {p['eeg_label']:<10} {p['esg_label']:<10} {p['emg_label']:<10} "
                f"{p['attn_eeg_esg']:<12.5f} {p['attn_esg_emg']:<12.5f} {p['chain_score']:<12.7f}"
            )
        print()

    def export_csv(self, pathways: List[Dict], path: str) -> None:
        """Write pathway table to CSV."""
        if not pathways:
            print(f"[PathwayExtractor] No pathways to export.")
            return
        keys = list(pathways[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(pathways)
        print(f"[PathwayExtractor] Saved {len(pathways)} pathways -> {path}")

    def reset(self) -> None:
        """Clear accumulated attention sums."""
        self._attn_sum.clear()
        self._attn_count.clear()
