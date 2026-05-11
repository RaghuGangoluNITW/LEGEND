#!/usr/bin/env python3
"""
Leave-One-Subject-Out (LOSO) training script for Lorentz TCN.

This script runs LOSO cross-validation on the Steele dataset,
training on N-1 subjects and validating on the held-out subject.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lorentz_tcnet.config import load_config
from lorentz_tcnet.train import run_loso_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Train Lorentz TCN with LOSO cross-validation"
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=None,
        help='Override max epochs (for quick testing)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=None,
        help='Override data directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Override device (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Apply overrides
    if args.max_epochs is not None:
        config.num_epochs = args.max_epochs
        print(f"  Override: num_epochs = {args.max_epochs}")
    
    if args.data_dir is not None:
        config.data_root = str(args.data_dir.parent)
        config.dataset_name = args.data_dir.name
        print(f"  Override: data_dir = {args.data_dir}")
    
    if args.device is not None:
        config.device = args.device
        print(f"  Override: device = {args.device}")
    
    print("\n" + "="*80)
    print("Starting LOSO Training")
    print("="*80)
    
    # Run experiment
    results = run_loso_experiment(config)
    
    # Print summary
    print("\n" + "="*80)
    print("LOSO Cross-Validation Results")
    print("="*80)
    
    for subject_id, metrics in results.items():
        print(f"\n{subject_id}:")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Precision: {metrics['precision']:.2f}%")
        print(f"  Recall: {metrics['recall']:.2f}%")
        print(f"  F1-Score: {metrics['f1']:.2f}%")
    
    # Compute mean metrics
    mean_acc = sum(m['accuracy'] for m in results.values()) / len(results)
    mean_prec = sum(m['precision'] for m in results.values()) / len(results)
    mean_rec = sum(m['recall'] for m in results.values()) / len(results)
    mean_f1 = sum(m['f1'] for m in results.values()) / len(results)
    
    print("\n" + "-"*80)
    print("Mean Performance:")
    print(f"  Accuracy: {mean_acc:.2f}%")
    print(f"  Precision: {mean_prec:.2f}%")
    print(f"  Recall: {mean_rec:.2f}%")
    print(f"  F1-Score: {mean_f1:.2f}%")
    print("="*80)


if __name__ == '__main__':
    main()
