"""
HyperLorentzNet-HGCN: Hybrid Hyperbolic Temporal + Graph Model.

Architecture
============

Stage 1 – Temporal Encoder (HyperLorentzNet spine, unchanged)
  EEG / ESG / EMG raw signals
      → per-modality ModalityEncoder (TCN + linear projection)
      → Lorentz projection π(z) = [√(1+||z||²) ; z]
  Output: hyperbolic node features for each channel group.

Stage 2 – Lightweight Hyperbolic Graph Head (Brain-HGCN style)
  Per-channel Lorentz features (N = n_eeg + n_esg + n_emg nodes)
      → 1–2 × LorentzGraphAttentionLayer (signed aggregation)
      → Fréchet mean pooling → trial embedding
      → MLP classifier

  Static graph: precomputed PLV-based tri-layer connectivity
                (built once from training data, stored as edge_index / edge_weight).

Together this gives:
  (A) Geometric excellence  – full Lorentz manifold operations
  (B) O(d) efficiency       – TCN Stage 1 unchanged, GNN Stage 2 tiny (N≈51)
  (C) Structural discovery  – attention weights → EEG→ESG→EMG pathway map

Usage
-----
    from src.lorentz_tcnet.model_hybrid import HyperLorentzNetHGCN

    model = HyperLorentzNetHGCN(
        eeg_channels=28, esg_channels=15, emg_channels=8,
        hidden_dim=64, latent_dim=32, num_classes=4,
        gnn_hidden=32, gnn_layers=1, gnn_heads=1,
        dropout=0.2,
    )

    # Pre-register the static graph (computed once from training data)
    model.register_graph(edge_index, edge_weight)

    # Forward pass
    out = model(eeg, esg, emg)
    # out['logits']       : (B, num_classes)
    # out['attn_weights'] : (E,)  — for pathway extraction
    # out['stage1']       : dict  — outputs from TriModalLorentzNet stage
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .model import TriModalLorentzNet, LorentzProjection
from .gnn_head import HyperbolicGraphHead


class HyperLorentzNetHGCN(nn.Module):
    """
    Two-stage hybrid: Lorentzian TCN encoder + hyperbolic graph attention head.

    Parameters
    ----------
    eeg_channels, esg_channels, emg_channels : int
        Channel counts per modality.
    hidden_dim : int
        TCN hidden dimension (Stage 1).
    latent_dim : int
        Spatial dimension of Lorentz-projected features (Stage 1 output d).
        Node features passed to Stage 2 have shape latent_dim+1.
    num_classes : int
    gnn_hidden : int
        Hidden spatial dimension of the hyperbolic GNN.
    gnn_layers : int
        Number of LorentzGraphAttentionLayer layers (1–2 recommended).
    gnn_heads : int
        Attention heads per GNN layer.
    dropout : float
    use_stage1_logits : bool
        If True, final logits = average of Stage 1 + Stage 2 classifiers
        (ensemble mode). If False, only Stage 2 logits are used.
    """

    def __init__(
        self,
        eeg_channels: int,
        esg_channels: int,
        emg_channels: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_classes: int = 4,
        gnn_hidden: int = 32,
        gnn_layers: int = 1,
        gnn_heads: int = 1,
        dropout: float = 0.2,
        use_stage1_logits: bool = True,
        t_stride: int = 4,
    ):
        super().__init__()

        self.n_eeg = eeg_channels
        self.n_esg = esg_channels
        self.n_emg = emg_channels
        self.n_nodes = eeg_channels + esg_channels + emg_channels
        self.latent_dim = latent_dim
        self.use_stage1_logits = use_stage1_logits
        self.t_stride = t_stride  # temporal downsampling (T=1000 -> T//t_stride)

        # ---- Stage 1: TriModalLorentzNet (keep exactly as-is) ----
        self.stage1 = TriModalLorentzNet(
            eeg_channels=eeg_channels,
            esg_channels=esg_channels,
            emg_channels=emg_channels,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        # ---- Stage 2: Hyperbolic Graph Head ----
        # Node feature dimension = latent_dim + 1  (Lorentz-projected)
        node_dim = latent_dim + 1
        self.stage2 = HyperbolicGraphHead(
            in_dim=node_dim,
            hidden_dim=gnn_hidden,
            out_dim=gnn_hidden,
            num_classes=num_classes,
            num_layers=gnn_layers,
            num_heads=gnn_heads,
            dropout=dropout,
            proto_dim=latent_dim,  # prototype dim = latent_dim (same feature space)
        )

        # Per-channel Lorentz projectors for EEG/ESG/EMG
        # These project per-channel TCN features into hyperbolic space
        self.lorentz_proj = LorentzProjection(latent_dim)

        # Channel-level encoders (shallow): extract one vector per channel
        # We reuse the conv stem + pool logic from ModalityEncoder
        self._ch_enc_eeg = _ChannelEncoder(eeg_channels, latent_dim)
        self._ch_enc_esg = _ChannelEncoder(esg_channels, latent_dim)
        self._ch_enc_emg = _ChannelEncoder(emg_channels, latent_dim)

        # Static graph buffers (registered after init)
        self.register_buffer("edge_index", torch.zeros(2, 0, dtype=torch.long))
        self.register_buffer("edge_weight", torch.zeros(0, dtype=torch.float32))
        self._graph_registered = False

        # Ensemble weight (learnable mix of stage1 + stage2 logits)
        if use_stage1_logits:
            self.ensemble_w = nn.Parameter(torch.tensor(0.5))

    # ------------------------------------------------------------------
    # Graph registration
    # ------------------------------------------------------------------

    def register_graph(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> None:
        """
        Store the static tri-layer graph.
        Call once before training, e.g.:
            builder = TriLayerGraphBuilder(28, 15, 8)
            ei, ew = builder.build(eeg_data, esg_data, emg_data)
            model.register_graph(ei, ew)
        """
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self._graph_registered = True
        n_nodes = self.n_nodes
        print(
            f"[HyperLorentzNetHGCN] Graph registered: "
            f"{n_nodes} nodes, {edge_index.shape[1]} directed edges"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        eeg: torch.Tensor,
        esg: Optional[torch.Tensor],
        emg: Optional[torch.Tensor],
        true_label: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        eeg        : (B, n_eeg, T)
        esg        : (B, n_esg, T) or None
        emg        : (B, n_emg, T) or None
        true_label : (B,) int, optional — supply during pathway extraction to
                     enable hard one-hot prototype conditioning per movement class

        Returns
        -------
        dict with:
            'logits'         : (B, num_classes)
            'stage1_logits'  : (B, num_classes)   from TriModalLorentzNet
            'stage2_logits'  : (B, num_classes)   from HyperbolicGraphHead
            'attn_weights'   : (B, E)
            'node_feats'     : (B, N, latent_dim+1)
            'graph_embed'    : (B, gnn_hidden+1)
        """
        B = eeg.shape[0]

        # ---- Temporal downsampling (applied to all modalities before both stages) ----
        if self.t_stride > 1:
            eeg = eeg[:, :, ::self.t_stride]
            if esg is not None:
                esg = esg[:, :, ::self.t_stride]
            if emg is not None:
                emg = emg[:, :, ::self.t_stride]

        # ---- Stage 1: Global trial embedding ----
        s1_out = self.stage1(eeg, esg, emg)
        stage1_logits = s1_out["logits"]           # (B, C)

        if not self._graph_registered:
            # Fallback: stage 1 only
            return {"logits": stage1_logits, "stage1_logits": stage1_logits,
                    "stage2_logits": stage1_logits, "attn_weights": None,
                    "node_feats": None, "graph_embed": None}

        # ---- Build per-channel node features ----
        # Per-channel encoders output (B, n_ch, latent_dim)
        eeg_ch = self._ch_enc_eeg(eeg)              # (B, n_eeg, latent_dim)
        if esg is not None:
            esg_ch = self._ch_enc_esg(esg)
        else:
            esg_ch = torch.zeros(B, self.n_esg, self.latent_dim, device=eeg.device)

        if emg is not None:
            emg_ch = self._ch_enc_emg(emg)
        else:
            emg_ch = torch.zeros(B, self.n_emg, self.latent_dim, device=eeg.device)

        # Lorentz project each channel vector
        def _proj_channels(ch_feats):
            # ch_feats: (B, n_ch, latent_dim)
            spatial = ch_feats
            time = torch.sqrt(1.0 + (spatial ** 2).sum(-1, keepdim=True) + 1e-6)
            return torch.cat([time, spatial], dim=-1)   # (B, n_ch, latent_dim+1)

        eeg_h = _proj_channels(eeg_ch)
        esg_h = _proj_channels(esg_ch)
        emg_h = _proj_channels(emg_ch)

        # Concatenate along node dimension: (B, N, latent_dim+1)
        node_feats = torch.cat([eeg_h, esg_h, emg_h], dim=1)

        # ---- Stage 2: Hyperbolic GNN (conditioned on Stage 1 class predictions) ----
        # During training:           soft Stage 1 logits condition the GNN attention
        # During pathway extraction: hard true_label one-hot gives class-pure prototypes
        s2_out = self.stage2(node_feats, self.edge_index, self.edge_weight,
                             stage1_logits=stage1_logits.detach(),
                             true_label=true_label)
        stage2_logits = s2_out["logits"]           # (B, C)

        # ---- Ensemble ----
        if self.use_stage1_logits:
            w = torch.sigmoid(self.ensemble_w)
            logits = w * stage1_logits + (1 - w) * stage2_logits
        else:
            logits = stage2_logits

        return {
            "logits": logits,
            "stage1_logits": stage1_logits,
            "stage2_logits": stage2_logits,
            "attn_weights": s2_out["attn_weights"],
            "node_feats": node_feats,
            "graph_embed": s2_out["graph_embed"],
        }

    # ------------------------------------------------------------------
    # Convenience: get node→modality mapping
    # ------------------------------------------------------------------

    def modality_of(self, node_idx: int) -> tuple[str, int]:
        """Return (modality_name, local_channel_index) for a global node index."""
        if node_idx < self.n_eeg:
            return "EEG", node_idx
        elif node_idx < self.n_eeg + self.n_esg:
            return "ESG", node_idx - self.n_eeg
        else:
            return "EMG", node_idx - self.n_eeg - self.n_esg


# ---------------------------------------------------------------------------
# Per-channel encoder (shallow: 1×conv + global average pool)
# ---------------------------------------------------------------------------

class _ChannelEncoder(nn.Module):
    """
    Lightweight encoder that maps each channel's time series to a latent vector.

    Input : (B, n_ch, T)
    Output: (B, n_ch, latent_dim)

    t_stride: temporal downsampling factor applied before Conv1d (default=4).
              Reduces T=1000 → 250, giving ~4× compute speedup with minimal
              accuracy loss for movement classification.
    """

    def __init__(self, n_channels: int, latent_dim: int,
                 kernel_size: int = 5, t_stride: int = 4):
        super().__init__()
        hidden = max(latent_dim, 16)
        self.t_stride = t_stride
        # Depthwise: process each channel independently
        self.conv1 = nn.Conv1d(1, hidden, kernel_size, padding=kernel_size // 2,
                               groups=1, bias=False)
        self.bn = nn.BatchNorm1d(hidden)
        self.proj = nn.Linear(hidden, latent_dim)
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_ch, T) → (B, n_ch, latent_dim)"""
        B, n_ch, T = x.shape
        # Temporal downsampling (strided slice): T → T//t_stride
        if self.t_stride > 1:
            x = x[:, :, ::self.t_stride]
        # Reshape to (B*n_ch, 1, T_ds) — process each channel independently
        x_flat = x.reshape(B * n_ch, 1, x.shape[-1])
        h = torch.relu(self.bn(self.conv1(x_flat)))   # (B*n_ch, hidden, T_ds)
        h = h.mean(-1)                                  # (B*n_ch, hidden)
        h = self.proj(h)                                # (B*n_ch, latent_dim)
        return h.reshape(B, n_ch, -1)                  # (B, n_ch, latent_dim)

