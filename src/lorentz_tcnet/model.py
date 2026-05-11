from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
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


class ModalityEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, latent_dim: int, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.tcn1 = TemporalBlock(hidden_dim, hidden_dim, dilation=1, dropout=dropout)
        self.tcn2 = TemporalBlock(hidden_dim, hidden_dim, dilation=2, dropout=dropout)
        self.tcn3 = TemporalBlock(hidden_dim, hidden_dim, dilation=4, dropout=dropout)
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem(x))
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.tcn3(x)
        x = x.mean(dim=-1)
        return self.proj(x)


class LorentzProjection(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x
        time = torch.sqrt(1.0 + torch.sum(spatial * spatial, dim=-1, keepdim=True) + 1e-6)
        return torch.cat([time, spatial], dim=-1)


def lorentz_inner(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return -(u[..., :1] * v[..., :1]) + torch.sum(u[..., 1:] * v[..., 1:], dim=-1, keepdim=True)


def lorentz_distance(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    ip = -lorentz_inner(u, v)
    ip = torch.clamp(ip, min=1.0 + 1e-6)
    return torch.acosh(ip)


class TriModalLorentzNet(nn.Module):
    def __init__(self, eeg_channels: int, esg_channels: int, emg_channels: int, hidden_dim: int, latent_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.eeg_encoder = ModalityEncoder(eeg_channels, hidden_dim, latent_dim, dropout)
        self.esg_encoder = ModalityEncoder(esg_channels, hidden_dim, latent_dim, dropout)
        self.emg_encoder = ModalityEncoder(emg_channels, hidden_dim, latent_dim, dropout)

        self.lorentz_proj = LorentzProjection(latent_dim)

        self.fusion = nn.Sequential(
            nn.Linear((latent_dim + 1) * 3 + 3, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, eeg: torch.Tensor, esg: torch.Tensor, emg: torch.Tensor) -> Dict[str, torch.Tensor]:
        eeg_z = self.eeg_encoder(eeg)
        
        # Handle optional modalities (for EEG-only mode)
        if esg is not None:
            esg_z = self.esg_encoder(esg)
            esg_h = self.lorentz_proj(esg_z)
        else:
            # Create dummy ESG embedding with zeros
            esg_z = torch.zeros_like(eeg_z)
            esg_h = self.lorentz_proj(esg_z)
        
        if emg is not None:
            emg_z = self.emg_encoder(emg)
            emg_h = self.lorentz_proj(emg_z)
        else:
            # Create dummy EMG embedding with zeros
            emg_z = torch.zeros_like(eeg_z)
            emg_h = self.lorentz_proj(emg_z)

        eeg_h = self.lorentz_proj(eeg_z)

        d_eeg_esg = lorentz_distance(eeg_h, esg_h)
        d_eeg_emg = lorentz_distance(eeg_h, emg_h)
        d_esg_emg = lorentz_distance(esg_h, emg_h)

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
