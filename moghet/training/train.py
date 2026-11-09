#!/usr/bin/env python
"""
MOGHET Unified Training Script (sequential single-device CV).

Runs all folds one after another on one GPU/CPU.
For 4-GPU fold-parallel training, use ``train_multi_gpu.py`` instead.

Both scripts share fold logic via ``training.fold_train`` (DRY).

  --verbosity quiet|normal|verbose|debug
  --resume   reuse cv_splits.json, skip finished folds, continue from checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

import torch

current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training.fold_train import (
    aggregate_cv_results,
    build_config,
    load_aligned_patient_ids,
    load_cv_splits,
    prepare_cv_splits,
    save_fold_splits,
    set_seed,
    train_one_fold,
)
from training.verbosity import log, parse_verbosity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOGHET Unified Training Script")
    parser.add_argument(
        "--dataset",
        type=str,
        default="BRCA",
        choices=[
            "BRCA", "LUAD", "COAD", "GBM", "KIRC", "LIHC", "LUNG", "OV", "SKCM",
            "LUSC", "STAD", "UCEC", "HNSC", "PAAD", "LGG", "PRAD",
        ],
        help="Dataset to use",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--smoke-test", action="store_true", help="Quick test mode")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=0,
        help="DataLoader num_workers (default 0; keep low to avoid hangs)",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="model",
        help="Results dir suffix: results/<dataset>_<suffix>",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="normal",
        help="quiet|normal|verbose|debug (or 0|1|2|3)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted run from checkpoints / skip completed folds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbosity = parse_verbosity(args.verbosity)
    set_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log(verbosity, "=" * 70, min_level=0)
    log(verbosity, f"  MOGHET Training Script - {args.dataset}", min_level=0)
    log(verbosity, "=" * 70, min_level=0)
    log(verbosity, f"Device: {device}", min_level=0)
    log(verbosity, f"Seed: {args.seed}", min_level=0)
    log(verbosity, f"Folds: {2 if args.smoke_test else args.n_folds}", min_level=0)
    log(verbosity, f"Verbosity: {args.verbosity} ({verbosity})", min_level=0)
    log(verbosity, f"Resume: {args.resume}", min_level=0)
    if args.smoke_test:
        log(verbosity, "Smoke Test Mode: ON", min_level=0)
    log(verbosity, "=" * 70, min_level=0)

    config = build_config(
        project_root,
        args.dataset,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_folds=args.n_folds,
        patience=args.patience,
        smoke_test=args.smoke_test,
        output_suffix=args.output_suffix,
    )
    os.makedirs(config["output_dir"], exist_ok=True)
    log(verbosity, f"Output directory: {config['output_dir']}", min_level=0)

    all_patient_ids, _, _ = load_aligned_patient_ids(config)
    log(verbosity, f"Valid patients: {len(all_patient_ids)}", min_level=0)

    schedule_path = osp.join(config["output_dir"], "cv_splits.json")
    fold_splits = None
    if args.resume:
        fold_splits = load_cv_splits(config["output_dir"])
        if fold_splits is not None:
            log(verbosity, f"Loaded existing CV splits from {schedule_path}", min_level=0)
        else:
            log(
                verbosity,
                "Warning: --resume set but cv_splits.json missing; creating new splits",
                min_level=0,
            )

    if fold_splits is None:
        _, fold_splits = prepare_cv_splits(all_patient_ids, config)
        save_fold_splits(config["output_dir"], fold_splits)
        with open(schedule_path, "w") as f:
            json.dump(fold_splits, f, indent=2)
    else:
        save_fold_splits(config["output_dir"], fold_splits)

    for split in fold_splits:
        train_one_fold(
            fold=split["fold"],
            train_patient_ids=split["train_patient_ids"],
            val_patient_ids=split["val_patient_ids"],
            test_patient_ids=split["test_patient_ids"],
            config=config,
            device=device,
            num_workers=args.dataloader_workers,
            resume=args.resume,
            verbosity=verbosity,
        )

    summary = aggregate_cv_results(
        output_dir=config["output_dir"],
        n_splits=config["n_splits"],
        dataset=args.dataset,
        seed=args.seed,
        config=config,
    )
    cv = summary["cross_validation_summary"]
    log(verbosity, "=" * 70, min_level=0)
    log(verbosity, "  Cross-validation Completed", min_level=0)
    log(verbosity, "=" * 70, min_level=0)
    log(
        verbosity,
        f"Average Test C-Index: {cv['mean_test_c_index']:.4f} ± {cv['std_test_c_index']:.4f}",
        min_level=0,
    )
    log(
        verbosity,
        f"Average Test AUC: {cv['mean_test_auc']:.4f} ± {cv['std_test_auc']:.4f}",
        min_level=0,
    )
    log(verbosity, f"Results saved at: {config['output_dir']}", min_level=0)


if __name__ == "__main__":
    main()
