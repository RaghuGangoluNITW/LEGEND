"""
Phase 4 Improved Lorentzian Tri-Modal Network
==============================================

TARGETED IMPROVEMENTS over Phase 3:
1. Conservative augmentation (0.5 → 0.15 probability)
2. Learnable modality fusion weights
3. Lorentzian-aware dropout
4. Better attention placement
5. Gradient clipping & learning rate scheduling

PRESERVES: Core Lorentzian geometric framework (NOVEL CONTRIBUTION)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

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
# PHASE 4 IMPROVEMENTS (NEW - TARGETED ENHANCEMENTS)
# ============================================================================

class LorentzianDropout(nn.Module):
    """
    Dropout that respects Lorentzian geometry.
    Drops entire embeddings rather than individual dimensions.
    """
    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
            
        # Drop entire embeddings (batch dimension)
        # x shape: [batch, lorentz_dim+1]
        batch_size = x.shape[0]
        mask = torch.rand(batch_size, 1, device=x.device) > self.p
        mask = mask.float() / (1 - self.p)  # Rescale
        return x * mask


class LearnableModalityFusion(nn.Module):
    """
    Learns importance weights for each modality distance.
    Adapts to subject-specific modality contributions.
    """
    def __init__(self):
        super().__init__()
        # Initialize with equal weights
        self.weights = nn.Parameter(torch.ones(3) / 3.0)
        
    def forward(self, d_eeg_esg: torch.Tensor, d_eeg_emg: torch.Tensor, 
                d_esg_emg: torch.Tensor) -> torch.Tensor:
        """
        Fuses three distance features with learned weights.
        Uses softmax for normalization.
        """
        w = F.softmax(self.weights, dim=0)
        
        # Weighted sum of distances
        d_combined = (w[0] * d_eeg_esg + 
                     w[1] * d_eeg_emg + 
                     w[2] * d_esg_emg)
        
        return d_combined


class SpatialAttention(nn.Module):
    """Channel-wise attention for spatial importance."""
    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time]
        gap = x.mean(dim=2)  # Global average pooling
        weights = self.fc(gap).unsqueeze(2)
        return x * weights


class TemporalConvNet(nn.Module):
    """Multi-scale temporal convolutional network."""
    def __init__(self, in_channels: int, hidden_channels: int, 
                 kernel_size: int = 3, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            
            layers.append(nn.Conv1d(
                in_channels if i == 0 else hidden_channels,
                hidden_channels,
                kernel_size,
                padding=padding,
                dilation=dilation
            ))
            layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImprovedModalityEncoder(nn.Module):
    """
    Enhanced modality encoder with spatial attention and TCN.
    """
    def __init__(self, in_channels: int, hidden_channels: int, 
                 latent_dim: int, use_spatial_attention: bool = True):
        super().__init__()
        
        self.spatial_attention = SpatialAttention(in_channels) if use_spatial_attention else None
        self.tcn = TemporalConvNet(in_channels, hidden_channels, num_layers=3, dropout=0.2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels, latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time]
        if self.spatial_attention is not None:
            x = self.spatial_attention(x)
            
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x


class ImprovedTriModalLorentzNet(nn.Module):
    """
    Phase 4 improved tri-modal Lorentzian network.
    
    IMPROVEMENTS over Phase 3:
    - Learnable modality fusion weights
    - Lorentzian-aware dropout
    - Better regularization
    - Spatial attention
    
    PRESERVED: Core Lorentzian geometry
    """
    def __init__(
        self,
        eeg_channels: int = 64,
        esg_channels: int = 16,
        emg_channels: int = 8,
        hidden_channels: int = 64,
        latent_dim: int = 32,
        num_classes: int = 4,
        use_spatial_attention: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Modality encoders
        self.eeg_encoder = ImprovedModalityEncoder(
            eeg_channels, hidden_channels, latent_dim, use_spatial_attention
        )
        self.esg_encoder = ImprovedModalityEncoder(
            esg_channels, hidden_channels, latent_dim, use_spatial_attention
        )
        self.emg_encoder = ImprovedModalityEncoder(
            emg_channels, hidden_channels, latent_dim, use_spatial_attention
        )
        
        # CORE: Lorentzian projection (UNCHANGED)
        self.lorentz_proj = LorentzProjection(latent_dim)
        
        # PHASE 4: Lorentzian-aware dropout
        self.lorentz_dropout = LorentzianDropout(p=dropout)
        
        # PHASE 4: Learnable modality fusion
        self.modality_fusion = LearnableModalityFusion()
        
        # Enhanced fusion network (no BatchNorm to avoid batch size=1 issues)
        self.fusion = nn.Sequential(
            nn.Linear(3, latent_dim),  # 3 distances → latent_dim
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classifier
        self.classifier = nn.Linear(latent_dim // 2, num_classes)
        
    def forward(self, eeg: torch.Tensor, esg: Optional[torch.Tensor] = None, 
                emg: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through improved Lorentzian network.
        
        Args:
            eeg: [batch, channels, time]
            esg: [batch, channels, time] (optional)
            emg: [batch, channels, time] (optional)
            
        Returns:
            logits: [batch, num_classes]
        """
        # Encode modalities
        eeg_z = self.eeg_encoder(eeg)
        
        # Handle EEG-only mode (for PhysioNet, BCI)
        if esg is None and emg is None:
            esg_z = torch.zeros_like(eeg_z)
            emg_z = torch.zeros_like(eeg_z)
        else:
            esg_z = self.esg_encoder(esg) if esg is not None else torch.zeros_like(eeg_z)
            emg_z = self.emg_encoder(emg) if emg is not None else torch.zeros_like(eeg_z)
        
        # CORE: Project to Lorentz hyperboloid
        eeg_h = self.lorentz_proj(eeg_z)
        esg_h = self.lorentz_proj(esg_z)
        emg_h = self.lorentz_proj(emg_z)
        
        # PHASE 4: Apply Lorentzian dropout
        eeg_h = self.lorentz_dropout(eeg_h)
        esg_h = self.lorentz_dropout(esg_h)
        emg_h = self.lorentz_dropout(emg_h)
        
        # CORE: Compute hyperbolic distances
        d_eeg_esg = lorentz_distance(eeg_h, esg_h).squeeze(-1)
        d_eeg_emg = lorentz_distance(eeg_h, emg_h).squeeze(-1)
        d_esg_emg = lorentz_distance(esg_h, emg_h).squeeze(-1)
        
        # PHASE 4: Learnable modality fusion
        d_combined = self.modality_fusion(d_eeg_esg, d_eeg_emg, d_esg_emg)
        
        # Stack distances for fusion network
        distances = torch.stack([d_eeg_esg, d_eeg_emg, d_esg_emg], dim=1)
        
        # Fusion and classification
        fused = self.fusion(distances)
        logits = self.classifier(fused)
        
        return logits


class ConservativeAugmentation:
    """
    Conservative data augmentation (PHASE 4 FIX).
    Reduced aggressiveness to prevent catastrophic drops.
    """
    
    @staticmethod
    def time_shift(x: torch.Tensor, max_shift: int = 5) -> torch.Tensor:
        """Time shift with reduced range (±5 samples vs ±10)."""
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=-1)
    
    @staticmethod
    def amplitude_scale(x: torch.Tensor, scale_range: Tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
        """Amplitude scaling with reduced range ([0.9, 1.1] vs [0.8, 1.2])."""
        scale = torch.empty(1).uniform_(*scale_range).item()
        return x * scale
    
    @staticmethod
    def add_noise(x: torch.Tensor, noise_std: float = 0.005) -> torch.Tensor:
        """Add Gaussian noise with reduced std (0.005 vs 0.01)."""
        noise = torch.randn_like(x) * noise_std
        return x + noise
    
    @staticmethod
    def channel_dropout(x: torch.Tensor, drop_prob: float = 0.05) -> torch.Tensor:
        """Channel dropout with reduced probability (0.05 vs 0.1)."""
        mask = torch.rand(x.shape[0], 1, device=x.device) > drop_prob
        return x * mask.float()
    
    @staticmethod
    def apply_augmentations(x: torch.Tensor, aug_prob: float = 0.15) -> torch.Tensor:
        """
        Apply augmentations with CONSERVATIVE probability (0.15 vs 0.5).
        PHASE 4 FIX: Reduces over-augmentation that caused catastrophic drops.
        """
        if torch.rand(1).item() < aug_prob:
            x = ConservativeAugmentation.time_shift(x)
        if torch.rand(1).item() < aug_prob:
            x = ConservativeAugmentation.amplitude_scale(x)
        if torch.rand(1).item() < aug_prob:
            x = ConservativeAugmentation.add_noise(x)
        if torch.rand(1).item() < aug_prob:
            x = ConservativeAugmentation.channel_dropout(x)
        return x


def get_model_summary(model: nn.Module) -> Dict[str, int]:
    """Get summary of model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }
