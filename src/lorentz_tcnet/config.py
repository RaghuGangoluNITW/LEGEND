from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class TrainConfig:
    data_root: str = "data"
    dataset_name: str = "steele"
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 5e-4
    weight_decay: float = 5e-4
    latent_dim: int = 128
    hidden_dim: int = 64
    tcn_dropout: float = 0.3
    num_classes: int = 4
    eeg_channels: int = 28
    esg_channels: int = 15
    emg_channels: int = 8
    num_workers: int = 2
    seed: int = 42
    device: str = "cuda"
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    save_dir: str = "results"

    @property
    def dataset_dir(self) -> Path:
        return Path(self.data_root) / self.dataset_name


def load_config(config_path: str) -> TrainConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return TrainConfig(**raw)
