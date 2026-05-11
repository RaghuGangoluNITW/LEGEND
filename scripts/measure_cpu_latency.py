"""
R1.5 — CPU inference latency measurement for LEGEND.

Usage:
    python scripts/measure_cpu_latency.py \
        --checkpoint results/hybrid_loso_v7/fold_NIS001/best_model.pt \
        --n_runs 200

Outputs mean ± std latency (ms) per 1-second epoch on CPU, printed and
saved to results/cpu_latency.json.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lorentz_tcnet.model_hybrid import HyperLorentzNetHGCN

# Match defaults used during training
N_EEG, N_ESG, N_EMG = 28, 15, 8
N_NODES = N_EEG + N_ESG + N_EMG   # 51
SAMPLE_RATE = 250                   # Hz
EPOCH_SECONDS = 1
T_SAMPLES = SAMPLE_RATE * EPOCH_SECONDS  # 250 time points


def build_dummy_graph(n_nodes: int = N_NODES):
    """Fully-connected directed graph as a stand-in for the PLV graph."""
    src, dst = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.ones(edge_index.shape[1])
    return edge_index, edge_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pt saved by train_hybrid_loso.py")
    parser.add_argument("--n_runs", type=int, default=200,
                        help="Number of timed forward passes (after warmup)")
    parser.add_argument("--n_warmup", type=int, default=20,
                        help="Warmup passes (not timed)")
    parser.add_argument("--output", default="results/cpu_latency.json")
    # Model hyper-parameters (must match training defaults)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--gnn_hidden", type=int, default=32)
    parser.add_argument("--gnn_layers", type=int, default=1)
    parser.add_argument("--gnn_heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--t_stride", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cpu")

    # Build model
    model = HyperLorentzNetHGCN(
        eeg_channels=N_EEG,
        esg_channels=N_ESG,
        emg_channels=N_EMG,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_classes=args.num_classes,
        gnn_hidden=args.gnn_hidden,
        gnn_layers=args.gnn_layers,
        gnn_heads=args.gnn_heads,
        dropout=args.dropout,
        use_stage1_logits=True,
        t_stride=args.t_stride,
    )

    # Load checkpoint weights
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Register graph first so buffer sizes match before loading state dict
    edge_index = ckpt["edge_index"]
    edge_weight = ckpt["edge_weight"]
    model.register_graph(edge_index, edge_weight)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Separate inputs for each modality (batch=1, channels, time)
    eeg_x = torch.randn(1, N_EEG, T_SAMPLES)
    esg_x = torch.randn(1, N_ESG, T_SAMPLES)
    emg_x = torch.randn(1, N_EMG, T_SAMPLES)

    print(f"Model loaded from: {args.checkpoint}")
    print(f"EEG: {list(eeg_x.shape)}  ESG: {list(esg_x.shape)}  EMG: {list(emg_x.shape)}")
    print(f"Device: CPU")
    print(f"Warmup: {args.n_warmup} passes | Timed: {args.n_runs} passes")

    # Warmup
    with torch.no_grad():
        for _ in range(args.n_warmup):
            _ = model(eeg_x, esg_x, emg_x)

    # Timed runs
    latencies_ms = []
    with torch.no_grad():
        for _ in range(args.n_runs):
            t0 = time.perf_counter()
            _ = model(eeg_x, esg_x, emg_x)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms = np.array(latencies_ms)
    mean_ms = float(np.mean(latencies_ms))
    std_ms = float(np.std(latencies_ms))
    p50_ms = float(np.percentile(latencies_ms, 50))
    p95_ms = float(np.percentile(latencies_ms, 95))
    p99_ms = float(np.percentile(latencies_ms, 99))

    print(f"\n=== CPU Inference Latency (1-second epoch, batch=1) ===")
    print(f"  Mean ± Std : {mean_ms:.2f} ± {std_ms:.2f} ms")
    print(f"  Median     : {p50_ms:.2f} ms")
    print(f"  P95        : {p95_ms:.2f} ms")
    print(f"  P99        : {p99_ms:.2f} ms")
    print(f"  Min / Max  : {latencies_ms.min():.2f} / {latencies_ms.max():.2f} ms")

    result = {
        "checkpoint": args.checkpoint,
        "n_runs": args.n_runs,
        "input_shape": [1, N_NODES, T_SAMPLES],
        "device": "cpu",
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "min_ms": float(latencies_ms.min()),
        "max_ms": float(latencies_ms.max()),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to: {args.output}")


if __name__ == "__main__":
    main()
