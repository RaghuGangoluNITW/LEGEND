"""
Lightweight Hyperbolic Graph Attention Head (inspired by Brain-HGCN).

Architecture
------------
  Input  : node features on the hyperboloid ℍⁿ  (from TriModalLorentzNet Stage 1)
  Layer  : 1–2 × LorentzGraphAttentionLayer
             – Lorentz inner product as attention score
             – Signed aggregation (separate pos / neg neighbour softmax)
             – Log-map → linear → Exp-map update
  Pooling: approximate Fréchet mean (single Karcher step) → trial vector
  Output : latent vector on ℍⁿ + Euclidean classifier

All geometry uses the Lorentz (hyperboloid) model:
  ⟨u, v⟩_L = −u₀v₀ + Σᵢ uᵢvᵢ
  Lorentz norm: ⟨u, u⟩_L = −1  (points on the hyperboloid)
  Lorentz distance: d(u,v) = arccosh(−⟨u,v⟩_L)

No external geometric library required — pure PyTorch.

Usage
-----
    from src.lorentz_tcnet.gnn_head import HyperbolicGraphHead, LorentzGraphAttentionLayer

    head = HyperbolicGraphHead(
        in_dim=33,        # latent_dim + 1  (Lorentz-projected)
        hidden_dim=32,
        out_dim=32,
        num_classes=4,
        num_layers=1,
        num_heads=1,
        dropout=0.2,
    )

    # node_feats: (B, N, in_dim)  — Lorentz-projected, points on hyperboloid
    # edge_index: (2, E)          — from TriLayerGraphBuilder
    # edge_weight: (E,)           — signed PLV weights
    out = head(node_feats, edge_index, edge_weight)
    # out['logits']       : (B, num_classes)
    # out['attn_weights'] : (E,)  for pathway extraction
    # out['graph_embed']  : (B, out_dim+1) hyperbolic trial embedding
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Low-level Lorentz geometry helpers
# ---------------------------------------------------------------------------

EPS = 1e-6
MIN_NORM = 1e-15


def lorentz_inner(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Minkowski inner product ⟨u, v⟩_L = −u₀v₀ + Σuᵢvᵢ
    u, v: (..., d+1)  → (..., 1)
    """
    return -(u[..., :1] * v[..., :1]) + (u[..., 1:] * v[..., 1:]).sum(-1, keepdim=True)


def lorentz_dist(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Geodesic distance on ℍⁿ.  u, v: (..., d+1) → (...,)"""
    ip = torch.clamp(-lorentz_inner(u, v).squeeze(-1), min=1.0 + EPS)
    return torch.acosh(ip)


def lorentz_log_map(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Logarithmic map at base point x.
    Maps u (on ℍⁿ) → tangent vector at x.
    x, u: (..., d+1)
    """
    ip = lorentz_inner(x, u)                          # (..., 1)
    ip = torch.clamp(ip, max=-(1.0 + EPS))            # keep inside domain
    alpha = torch.acosh(torch.clamp(-ip, min=1.0 + EPS))
    coeff = alpha / torch.clamp(
        torch.sqrt(torch.clamp(ip ** 2 - 1, min=EPS)), min=MIN_NORM
    )
    return coeff * (u + ip * x)


def lorentz_exp_map(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Exponential map at base point x.
    Maps tangent vector v → point on ℍⁿ.
    x, v: (..., d+1)
    """
    vnorm = torch.clamp(
        torch.sqrt(torch.clamp(lorentz_inner(v, v), min=EPS)), min=MIN_NORM
    )
    return torch.cosh(vnorm) * x + torch.sinh(vnorm) * v / vnorm


def lorentz_project(x: torch.Tensor) -> torch.Tensor:
    """
    Project a Euclidean vector z ∈ ℝᵈ onto ℍⁿ:
      t = sqrt(1 + ||z||²),  result = [t; z]
    Accepts either (..., d) or (..., d+1) — normalises if d+1.
    """
    if True:  # always treat as spatial part
        spatial = x[..., 1:] if x.shape[-1] > 1 else x
        time = torch.sqrt(1.0 + (spatial ** 2).sum(-1, keepdim=True) + EPS)
        return torch.cat([time, spatial], dim=-1)


def frechet_mean_step(points: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Single Karcher / Fréchet mean step on the hyperboloid.
    points  : (..., N, d+1) — N points on ℍⁿ
    weights : (..., N)      — optional non-negative weights (summing to 1)
    Returns : (..., d+1)    — approximate Fréchet mean
    """
    if weights is None:
        mu = points.mean(dim=-2)                   # (..., d+1)
    else:
        mu = (weights.unsqueeze(-1) * points).sum(-2)  # (..., d+1)
    # Project back onto hyperboloid
    spatial = mu[..., 1:]
    time = torch.sqrt(1.0 + (spatial ** 2).sum(-1, keepdim=True) + EPS)
    return torch.cat([time, spatial], dim=-1)


# ---------------------------------------------------------------------------
# Lorentz Graph Attention Layer
# ---------------------------------------------------------------------------

class LorentzGraphAttentionLayer(nn.Module):
    """
    One layer of hyperbolic graph attention with signed aggregation.

    Algorithm
    ---------
    For each node i:
    1. Compute attention score with neighbour j:
         score_ij = LeakyReLU( a^T · [log_xᵢ(xᵢ) ‖ log_xᵢ(xⱼ)] )
       (using Lorentz log map, so all operations are tangent-space linear).
    2. Split neighbours into positive (edge_weight > 0) and negative (< 0).
    3. Separate softmax over each group → α⁺_ij, α⁻_ij.
    4. Lorentz-weighted aggregation in tangent space at xᵢ.
    5. Exp-map back to ℍⁿ.
    6. Residual add (using Möbius / simple tangent addition).

    Parameters
    ----------
    in_dim   : d+1  (Lorentz dimension; in_dim-1 is spatial dim)
    out_dim  : output spatial dimension (output Lorentz dim = out_dim+1)
    heads    : multi-head attention
    dropout  : attention dropout
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 1,
        dropout: float = 0.2,
        negative_slope: float = 0.2,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        self.d_head = out_dim // heads
        self.cond_dim = cond_dim

        # Linear projection: tangent space (in_dim-1) → out_dim
        self.W = nn.Linear(in_dim - 1, out_dim, bias=False)
        # Attention vector per head: concatenation of source & target
        self.attn_vec = nn.Parameter(torch.Tensor(1, heads, 2 * self.d_head))
        nn.init.xavier_uniform_(self.attn_vec.view(1, -1).unsqueeze(0))

        # Class-prototype conditioning: maps proto vector → per-head bias
        if cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, heads, bias=True)
            nn.init.zeros_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)
        else:
            self.cond_proj = None

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.attn_drop = nn.Dropout(dropout)

        # Lorentz projection after update
        # output shape: (..., out_dim + 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        cond_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x           : (N, in_dim)   node features on ℍⁿ
        edge_index  : (2, E)        [src; dst]
        edge_weight : (E,)          signed PLV weights
        cond_vec    : (cond_dim,)   optional class-prototype conditioning vector

        Returns
        -------
        out         : (N, out_dim+1)  updated node features on ℍⁿ
        attn_scores : (E,)            final attention values (for pathway extraction)
        """
        N = x.shape[0]
        src, dst = edge_index[0], edge_index[1]   # each (E,)

        # Project spatial features with shared W
        x_tan = x[:, 1:]                          # (N, in_dim-1) tangent
        x_proj = self.W(x_tan)                    # (N, out_dim)
        x_proj = x_proj.view(N, self.heads, self.d_head)  # (N, H, d_head)

        # Attention score: concat projected src & dst features
        src_feat = x_proj[src]                    # (E, H, d_head)
        dst_feat = x_proj[dst]                    # (E, H, d_head)
        alpha = torch.cat([src_feat, dst_feat], dim=-1)  # (E, H, 2*d_head)
        alpha = (self.attn_vec * alpha).sum(-1)   # (E, H)

        # Class-prototype bias: shifts attention distribution toward class-relevant edges
        if self.cond_proj is not None and cond_vec is not None:
            # cond_vec: (cond_dim,) → (1, H) broadcast over all edges
            proto_bias = self.cond_proj(cond_vec).view(1, self.heads)  # (1, H)
            alpha = alpha + proto_bias

        alpha = self.leaky_relu(alpha)            # (E, H)

        # PLV sign only: gates which neighbours are excitatory (+) or inhibitory (-).
        # Do NOT scale by PLV magnitude — that caused the magnitude to dominate softmax
        # and made all attention scores nearly identical regardless of input content.
        # The learned alpha scores carry all the input-dependent differentiation.

        # ---- Signed aggregation ----
        # Positive neighbours
        pos_mask = (edge_weight > 0).float().unsqueeze(-1)  # (E, 1)
        neg_mask = (edge_weight < 0).float().unsqueeze(-1)

        def _softmax_group(scores, mask, dst_idx, N):
            # Zero out other group, compute softmax over neighbours
            masked = scores * mask + (-1e9) * (1 - mask)  # (E, H)
            # Scatter softmax: for each dst node, softmax over its sources
            # Use manual scatter for compatibility (no pyg dependency)
            out_attn = torch.zeros(N, scores.shape[1], device=scores.device)
            # Shift to positive for numerical stability per node
            max_val = torch.zeros(N, scores.shape[1], device=scores.device)
            max_val.scatter_reduce_(0, dst_idx.unsqueeze(-1).expand_as(masked),
                                    masked, reduce="amax", include_self=True)
            exp_a = torch.exp(masked - max_val[dst_idx])  # (E, H)
            exp_a = exp_a * mask  # zero out other group
            sum_exp = torch.zeros(N, scores.shape[1], device=scores.device)
            sum_exp.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(exp_a), exp_a)
            norm_a = exp_a / (sum_exp[dst_idx] + 1e-9)  # (E, H)
            return norm_a

        attn_pos = _softmax_group(alpha, pos_mask, dst, N)   # (E, H)
        attn_neg = _softmax_group(alpha, neg_mask, dst, N)   # (E, H)
        attn_all = attn_pos - 0.5 * attn_neg                 # combined signed attn
        attn_all = self.attn_drop(attn_all)

        # ---- Aggregate in tangent space ----
        # Use x_proj as message, weighted by attn_all
        msg = x_proj[src] * attn_all.unsqueeze(-1)           # (E, H, d_head)
        msg = msg.view(msg.shape[0], self.out_dim)            # (E, out_dim)
        agg = torch.zeros(N, self.out_dim, device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)  # (N, out_dim)

        # ---- Map back to hyperboloid ----
        # Simple Lorentz projection of aggregated tangent features
        time = torch.sqrt(1.0 + (agg ** 2).sum(-1, keepdim=True) + EPS)
        out = torch.cat([time, agg], dim=-1)                  # (N, out_dim+1)

        # Summary attention score per edge (mean over heads, absolute value)
        attn_scores = attn_all.abs().mean(-1)                 # (E,)

        return out, attn_scores


# ---------------------------------------------------------------------------
# Full Hyperbolic Graph Head
# ---------------------------------------------------------------------------

class HyperbolicGraphHead(nn.Module):
    """
    Stage 2: lightweight hyperbolic GNN head for classification + pathway extraction.

    Input
    -----
    node_feats  : (B, N, in_dim)   — Lorentz-projected per-channel embeddings
                                      in_dim = latent_dim + 1
    edge_index  : (2, E)           — static, precomputed from TriLayerGraphBuilder
    edge_weight : (E,)             — signed PLV weights

    Output
    ------
    dict with:
        'logits'       : (B, num_classes)
        'attn_weights' : (E,)  — averaged attention (for pathway discovery)
        'graph_embed'  : (B, out_dim+1)  — Fréchet-pooled trial embedding
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_classes: int,
        num_layers: int = 1,
        num_heads: int = 1,
        dropout: float = 0.2,
        proto_dim: int = 32,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.proto_dim = proto_dim

        # Learnable class prototypes in Euclidean space (proto_dim)
        # Conditioned via soft mixture of Stage 1 class probabilities
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, proto_dim) * 0.01)

        dims = [in_dim] + [hidden_dim + 1] * (num_layers - 1) + [out_dim + 1]
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                LorentzGraphAttentionLayer(
                    in_dim=dims[i],
                    out_dim=hidden_dim if i < num_layers - 1 else out_dim,
                    heads=num_heads,
                    dropout=dropout,
                    cond_dim=proto_dim,
                )
            )

        # Euclidean classifier on top of Fréchet mean (spatial part only)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

        self._last_attn: Optional[torch.Tensor] = None

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        stage1_logits: Optional[torch.Tensor] = None,
        true_label: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        node_feats     : (B, N, in_dim)
        edge_index     : (2, E)
        edge_weight    : (E,)
        stage1_logits  : (B, num_classes) — soft Stage 1 logits for prototype conditioning
        true_label     : (B,) int — if provided, use hard one-hot prototype (for
                         pathway extraction). Overrides stage1_logits conditioning.
        """
        B, N, _ = node_feats.shape
        device = node_feats.device

        # Make sure edge tensors are on the right device
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)

        E = edge_index.shape[1]
        # Per-sample attention: (B, E) so pathway extractor can condition on class
        attn_per_sample = torch.zeros(B, E, device=device)

        # Compute class prototype for attention conditioning.
        # During training: soft mixture via Stage 1 softmax.
        # During pathway extraction (true_label provided): hard one-hot prototype
        #   so each sample is conditioned purely on its true movement class.
        if true_label is not None:
            # Hard one-hot: (B, C) with 1.0 at true class
            one_hot = F.one_hot(true_label.to(node_feats.device),
                                num_classes=self.num_classes).float()  # (B, C)
            proto_b = one_hot @ self.class_prototypes                   # (B, proto_dim)
        elif stage1_logits is not None:
            proto_weights = F.softmax(stage1_logits.detach(), dim=-1)   # (B, C)
            proto_b = proto_weights @ self.class_prototypes              # (B, proto_dim)
        else:
            proto_b = None

        # Process each sample (small graph → batch loop is fine for N≈51)
        E = edge_index.shape[1]
        outputs = []
        for b in range(B):
            h = node_feats[b]                      # (N, in_dim)
            cond_vec = proto_b[b] if proto_b is not None else None
            last_attn = None
            for layer in self.layers:
                h, attn_scores = layer(h, edge_index, edge_weight, cond_vec=cond_vec)
                last_attn = attn_scores.detach()   # keep last layer's attention
            outputs.append(h)                      # (N, out_dim+1)
            if last_attn is not None:
                attn_per_sample[b] = last_attn     # (E,)
            # else: gnn_layers=0 — leave attn_per_sample[b] as zeros

        node_out = torch.stack(outputs, dim=0)     # (B, N, out_dim+1)

        # Fréchet mean pooling → (B, out_dim+1)
        graph_embed = frechet_mean_step(node_out)  # (B, out_dim+1)

        # Classify from spatial part
        logits = self.classifier(graph_embed[:, 1:])  # (B, num_classes)

        self._last_attn = attn_per_sample.cpu()

        return {
            "logits": logits,
            "attn_weights": attn_per_sample,       # (B, E) — per-sample
            "graph_embed": graph_embed,
        }

    @property
    def last_attention(self) -> Optional[torch.Tensor]:
        """Return the most recently computed per-sample attention weights (B, E)."""
        return self._last_attn
