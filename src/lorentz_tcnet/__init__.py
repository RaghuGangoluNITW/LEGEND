from .config import TrainConfig, load_config
from .model import TriModalLorentzNet
from .model_hybrid import HyperLorentzNetHGCN
from .graph import TriLayerGraphBuilder
from .pathway import PathwayExtractor

def run_loso_experiment(*args, **kwargs):
    """Lazy import wrapper to avoid sklearn/numpy compatibility issues at package load."""
    from .train import run_loso_experiment as _fn
    return _fn(*args, **kwargs)

__all__ = [
    "TrainConfig", "load_config",
    "TriModalLorentzNet",
    "HyperLorentzNetHGCN",
    "TriLayerGraphBuilder",
    "PathwayExtractor",
    "run_loso_experiment",
]
