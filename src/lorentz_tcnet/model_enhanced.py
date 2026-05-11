"""
Phase 3 Enhanced Lorentzian Tri-Modal Network
==============================================

CRITICAL: This module PRESERVES the core Lorentzian geometric framework while adding
complementary enhancements that operate AFTER Lorentzian feature extraction.

Core Lorentzian Components (UNCHANGED):
- LorentzProjection: Hyperboloid embedding with timelike coordinate
- lorentz_inner: Minkowski inner product with signature (-, +, +, ...)
- lorentz_distance: Hyperbolic distance computation
- Tri-modal connectivity: d_eeg_esg, d_eeg_emg, d_esg_emg

Phase 3 Enhancements (NEW - operate on TOP of Lorentzian features):
- Multi-head attention over Lorentzian embeddings
- Subject-adaptive batch normalization
- Spatial attention for channel importance
- Enhanced temporal modeling
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# CORE LORENTZIAN GEOMETRY (UNCHANGED - NOVEL CONTRIBUTION)
# ============================================================================

class LorentzProjection(nn.Module):
    """
    Projects Euclidean embeddings to Lorentz hyperboloid.
    CORE INNOVATION - DO NOT MODIFY
    
    Input: z ∈ R^d (Euclidean space)
    Output: h = [√(1+||z||²), z] ∈ H^d (Lorentz hyperboloid)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x
        # Timelike coordinate ensures point lies on hyperboloid
        time = torch.sqrt(1.0 + torch.sum(spatial * spatial, dim=-1, keepdim=True) + 1e-6)
        return torch.cat([time, spatial], dim=-1)


def lorentz_inner(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Minkowski inner product with signature (-, +, +, ..., +).
    CORE INNOVATION - DO NOT MODIFY
    
    ⟨u,v⟩_L = -u₀v₀ + Σᵢ uᵢvᵢ
    """
    return -(u[..., :1] * v[..., :1]) + torch.sum(u[..., 1:] * v[..., 1:], dim=-1, keepdim=True)


def lorentz_distance(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Hyperbolic distance between points on Lorentz hyperboloid.
    CORE INNOVATION - DO NOT MODIFY
    
    d(u,v) = acosh(-⟨u,v⟩_L)
    """
    ip = -lorentz_inner(u, v)
    ip = torch.clamp(ip, min=1.0 + 1e-6)
    return torch.acosh(ip)


# ============================================================================
# PHASE 3 ENHANCEMENTS (NEW - COMPLEMENTARY MODULES)
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention over Lorentzian embeddings.
    ENHANCEMENT: Learns which Lorentzian features are most discriminative.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, dim) - Lorentzian features
        Returns:
            attended: (batch, dim)
        """
        B = x.shape[0]
        
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # (B, 1, dim)
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, 1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, 1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = (attn @ v).transpose(1, 2).reshape(B, 1, self.dim)
        out = self.proj(out)
        
        return out.squeeze(1)  # (B, dim)


class SpatialAttention(nn.Module):
    """
    Channel-wise attention for spatial (electrode) importance.
    ENHANCEMENT: Learns which EEG channels are most relevant.
    """
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)
        Returns:
            weighted: (batch, channels, time)
        """
        B, C, T = x.shape
        
        # Channel statistics
        avg_out = self.avg_pool(x).view(B, C)  # (B, C)
        max_out = self.max_pool(x).view(B, C)  # (B, C)
        
        # Channel attention weights
        avg_weight = self.fc(avg_out)  # (B, C)
        max_weight = self.fc(max_out)  # (B, C)
        
        weight = (avg_weight + max_weight).unsqueeze(-1)  # (B, C, 1)
        
        return x * weight


class SubjectAdaptiveBatchNorm(nn.Module):
    """
    Subject-specific batch normalization to handle inter-subject variability.
    ENHANCEMENT: Reduces domain shift between subjects in LOSO.
    """
    def __init__(self, num_features: int, num_subjects: int = 10, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.num_subjects = num_subjects
        
        # Shared parameters
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum)
        
        # Subject-specific affine parameters
        self.subject_gamma = nn.Parameter(torch.ones(num_subjects, num_features))
        self.subject_beta = nn.Parameter(torch.zeros(num_subjects, num_features))
        
    def forward(self, x: torch.Tensor, subject_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)
            subject_ids: (batch,) - subject indices, optional
        Returns:
            normalized: (batch, channels, time)
        """
        # Standard batch normalization
        x = self.bn(x)
        
        # Apply subject-specific affine if IDs provided
        if subject_ids is not None:
            gamma = self.subject_gamma[subject_ids]  # (B, C)
            beta = self.subject_beta[subject_ids]   # (B, C)
            x = x * gamma.unsqueeze(-1) + beta.unsqueeze(-1)
        
        return x


class TemporalBlock(nn.Module):
    """
    Temporal Convolutional Block (UNCHANGED from baseline).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop2 = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.residual is None else self.residual(x)
        out = self.conv1(x)
        out = out[:, :, : x.shape[-1]]
        out = self.drop1(F.relu(self.bn1(out)))
        out = self.conv2(out)
        out = out[:, :, : x.shape[-1]]
        out = self.drop2(F.relu(self.bn2(out)))
        return F.relu(out + residual)


class EnhancedModalityEncoder(nn.Module):
    """
    Enhanced modality encoder with spatial attention.
    ENHANCEMENT: Spatial attention added BEFORE Lorentzian projection.
    """
    def __init__(self, in_channels: int, hidden_dim: int, latent_dim: int, dropout: float = 0.2):
        super().__init__()
        
        # Spatial attention (NEW)
        self.spatial_attn = SpatialAttention(in_channels)
        
        # Original temporal processing (UNCHANGED)
        self.stem = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.tcn1 = TemporalBlock(hidden_dim, hidden_dim, dilation=1, dropout=dropout)
        self.tcn2 = TemporalBlock(hidden_dim, hidden_dim, dilation=2, dropout=dropout)
        self.tcn3 = TemporalBlock(hidden_dim, hidden_dim, dilation=4, dropout=dropout)
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)
        Returns:
            z: (batch, latent_dim)
        """
        # NEW: Apply spatial attention first
        x = self.spatial_attn(x)
        
        # UNCHANGED: Temporal processing
        x = F.relu(self.stem(x))
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.tcn3(x)
        x = x.mean(dim=-1)  # Global average pooling
        return self.proj(x)


# ============================================================================
# PHASE 3 ENHANCED MODEL (LORENTZIAN CORE PRESERVED)
# ============================================================================

class EnhancedTriModalLorentzNet(nn.Module):
    """
    Phase 3 Enhanced Tri-Modal Lorentzian Network.
    
    Architecture Flow:
    1. Input → Spatial Attention (NEW)
    2. → Temporal Encoding (UNCHANGED)
    3. → Lorentzian Projection (CORE - UNCHANGED)
    4. → Hyperbolic Distances (CORE - UNCHANGED)
    5. → Multi-Head Attention (NEW)
    6. → Fusion & Classification
    
    The Lorentzian geometric framework remains the CORE innovation.
    Enhancements are auxiliary improvements that complement it.
    """
    def __init__(
        self, 
        eeg_channels: int, 
        esg_channels: int, 
        emg_channels: int, 
        hidden_dim: int, 
        latent_dim: int, 
        num_classes: int, 
        dropout: float = 0.2,
        use_attention: bool = True
    ):
        super().__init__()
        
        # Enhanced modality encoders (with spatial attention)
        self.eeg_encoder = EnhancedModalityEncoder(eeg_channels, hidden_dim, latent_dim, dropout)
        self.esg_encoder = EnhancedModalityEncoder(esg_channels, hidden_dim, latent_dim, dropout)
        self.emg_encoder = EnhancedModalityEncoder(emg_channels, hidden_dim, latent_dim, dropout)

        # CORE LORENTZIAN PROJECTION (UNCHANGED)
        self.lorentz_proj = LorentzProjection(latent_dim)
        
        # NEW: Multi-head attention over Lorentzian embeddings
        self.use_attention = use_attention
        if use_attention:
            # Use attention on the latent_dim only (before Lorentz projection adds time coordinate)
            self.mha_eeg = MultiHeadAttention(latent_dim, num_heads=4, dropout=dropout)
            self.mha_esg = MultiHeadAttention(latent_dim, num_heads=4, dropout=dropout)
            self.mha_emg = MultiHeadAttention(latent_dim, num_heads=4, dropout=dropout)

        # Fusion and classification
        fusion_input_dim = (latent_dim + 1) * 3 + 3  # Lorentzian embeddings + distances
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, eeg: torch.Tensor, esg: torch.Tensor, emg: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass preserving Lorentzian core with enhancements.
        
        Args:
            eeg: (batch, eeg_channels, time)
            esg: (batch, esg_channels, time) or None
            emg: (batch, emg_channels, time) or None
            
        Returns:
            dict with 'logits', 'eeg_h', 'esg_h', 'emg_h', 'connectivity'
        """
        # Step 1: Encode modalities (with spatial attention)
        eeg_z = self.eeg_encoder(eeg)
        
        # Handle optional modalities (for EEG-only datasets)
        if esg is not None:
            esg_z = self.esg_encoder(esg)
        else:
            esg_z = torch.zeros_like(eeg_z)
        
        if emg is not None:
            emg_z = self.emg_encoder(emg)
        else:
            emg_z = torch.zeros_like(eeg_z)
        
        # Step 2: CORE LORENTZIAN PROJECTION (UNCHANGED)
        eeg_h = self.lorentz_proj(eeg_z)
        esg_h = self.lorentz_proj(esg_z)
        emg_h = self.lorentz_proj(emg_z)
        
        # Step 3: NEW - Apply attention to latent features BEFORE projection
        if self.use_attention:
            eeg_z = eeg_z + self.mha_eeg(eeg_z)  # Residual connection
            esg_z = esg_z + self.mha_esg(esg_z)
            emg_z = emg_z + self.mha_emg(emg_z)
            # Re-project after attention
            eeg_h = self.lorentz_proj(eeg_z)
            esg_h = self.lorentz_proj(esg_z)
            emg_h = self.lorentz_proj(emg_z)
        
        # Step 4: CORE HYPERBOLIC DISTANCES (UNCHANGED)
        d_eeg_esg = lorentz_distance(eeg_h, esg_h)
        d_eeg_emg = lorentz_distance(eeg_h, emg_h)
        d_esg_emg = lorentz_distance(esg_h, emg_h)

        # Step 5: Fusion and classification
        fused_input = torch.cat([eeg_h, esg_h, emg_h, d_eeg_esg, d_eeg_emg, d_esg_emg], dim=-1)
        fused = self.fusion(fused_input)
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "eeg_h": eeg_h,
            "esg_h": esg_h,
            "emg_h": emg_h,
            "connectivity": torch.cat([d_eeg_esg, d_eeg_emg, d_esg_emg], dim=-1),
        }


# ============================================================================
# DATA AUGMENTATION (NEW)
# ============================================================================

class TimeSeriesAugmentation:
    """
    Data augmentation strategies for EEG/physiological signals.
    ENHANCEMENT: Improves generalization without changing model architecture.
    """
    
    @staticmethod
    def time_shift(x: torch.Tensor, max_shift: int = 10) -> torch.Tensor:
        """Randomly shift signal in time."""
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        if shift > 0:
            x = torch.cat([x[:, :, shift:], x[:, :, :shift]], dim=2)
        elif shift < 0:
            x = torch.cat([x[:, :, shift:], x[:, :, :shift]], dim=2)
        return x
    
    @staticmethod
    def amplitude_scale(x: torch.Tensor, scale_range: tuple = (0.8, 1.2)) -> torch.Tensor:
        """Randomly scale amplitude."""
        scale = torch.empty(1).uniform_(*scale_range).item()
        return x * scale
    
    @staticmethod
    def gaussian_noise(x: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(x) * noise_level
        return x + noise
    
    @staticmethod
    def channel_dropout(x: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
        """Randomly drop channels."""
        B, C, T = x.shape
        mask = torch.rand(B, C, 1, device=x.device) > drop_prob
        return x * mask.float()
    
    @staticmethod
    def apply_augmentations(x: torch.Tensor, aug_prob: float = 0.5) -> torch.Tensor:
        """Apply random augmentations during training."""
        if torch.rand(1).item() < aug_prob:
            x = TimeSeriesAugmentation.time_shift(x)
        if torch.rand(1).item() < aug_prob:
            x = TimeSeriesAugmentation.amplitude_scale(x)
        if torch.rand(1).item() < aug_prob:
            x = TimeSeriesAugmentation.gaussian_noise(x)
        if torch.rand(1).item() < aug_prob * 0.5:  # Less frequent
            x = TimeSeriesAugmentation.channel_dropout(x)
        return x
