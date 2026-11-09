#!/usr/bin/env python
"""
Multi-GPU cross-validation trainer for PathMoG / MOGHET.

Strategy (Arrhenius-style node with up to 4 GPUs)
-------------------------------------------------
- Each CV fold trains on its own GPU (1 fold -> 1 GPU).
- Up to ``--num-gpus`` (default 4) folds run concurrently.
- When a GPU frees, the next pending fold (e.g. fold 5) starts on that GPU.
- No DistributedDataParallel: folds are independent processes.

Checkpoints / resume
--------------------
Each fold writes ``fold_N/checkpoint.pt`` every epoch and ``fold_result.json``
when finished. Re-run with ``--resume`` (and the same ``--output-suffix``) to:
- reuse ``cv_splits.json`` (same train/val/test IDs)
- skip completed folds
- continue incomplete folds from their checkpoint

Verbosity
---------
  --verbosity quiet|normal|verbose|debug   (or 0|1|2|3)
  Default: normal (one line per epoch; good for long Kaggle runs)

Examples
--------
  python -m training.train_multi_gpu --dataset BRCA --num-gpus 4
  python -m training.train_multi_gpu --dataset BRCA --num-gpus 2 --batch-size 6 \\
      --verbosity quiet --resume --patience 10
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import os.path as osp
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any, Deque, Dict, List

import torch

_CURRENT_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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


def _fold_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process entry point: bind one fold to one GPU and train (picklable)."""
    fold = int(payload["fold"])
    gpu_id = int(payload["gpu_id"])
    config = payload["config"]
    split = payload["split"]
    num_workers = int(payload.get("num_workers", 0))
    resume = bool(payload.get("resume", False))
    verbosity = int(payload.get("verbosity", 1))

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        log(verbosity, f"[Fold {fold}] CUDA unavailable; falling back to CPU", min_level=0)

    log(verbosity, f"[Worker] Starting fold {fold} on {device}", min_level=0)
    result = train_one_fold(
        fold=fold,
        train_patient_ids=split["train_patient_ids"],
        val_patient_ids=split["val_patient_ids"],
        test_patient_ids=split["test_patient_ids"],
        config=config,
        device=device,
        num_workers=num_workers,
        resume=resume,
        verbosity=verbosity,
    )
    result["gpu_id"] = gpu_id
    return result


def _available_cuda_count() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def run_fold_pool(
    fold_splits: List[Dict[str, Any]],
    config: Dict[str, Any],
    num_gpus: int,
    num_workers: int = 0,
    resume: bool = False,
    verbosity: int = 1,
) -> List[Dict[str, Any]]:
    """Schedule folds onto a pool of GPUs; resume/skip handled inside train_one_fold."""
    n_cuda = _available_cuda_count()
    if n_cuda == 0:
        log(verbosity, "No CUDA devices found; running folds sequentially on CPU.", min_level=0)
        results = []
        for split in fold_splits:
            results.append(
                train_one_fold(
                    fold=split["fold"],
                    train_patient_ids=split["train_patient_ids"],
                    val_patient_ids=split["val_patient_ids"],
                    test_patient_ids=split["test_patient_ids"],
                    config=config,
                    device=torch.device("cpu"),
                    num_workers=num_workers,
                    resume=resume,
                    verbosity=verbosity,
                )
            )
        return results

    num_gpus = max(1, min(num_gpus, n_cuda, len(fold_splits)))
    log(
        verbosity,
        f"Scheduling {len(fold_splits)} folds on {num_gpus} GPU(s) "
        f"(device_count={n_cuda}, resume={resume})",
        min_level=0,
    )

    pending_folds: Deque[Dict[str, Any]] = deque(fold_splits)
    free_gpus: Deque[int] = deque(range(num_gpus))
    results: List[Dict[str, Any]] = []

    ctx = mp.get_context("spawn")
    pending_futures: Dict[Any, int] = {}

    with ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:

        def _submit_available() -> None:
            while pending_folds and free_gpus:
                split = pending_folds.popleft()
                gpu_id = free_gpus.popleft()
                payload = {
                    "fold": split["fold"],
                    "gpu_id": gpu_id,
                    "config": config,
                    "split": split,
                    "num_workers": num_workers,
                    "resume": resume,
                    "verbosity": verbosity,
                }
                log(verbosity, f"Submit fold {split['fold']} -> GPU {gpu_id}", min_level=0)
                fut = executor.submit(_fold_worker, payload)
                pending_futures[fut] = gpu_id

        _submit_available()

        while pending_futures:
            done, _ = wait(list(pending_futures.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                gpu_id = pending_futures.pop(fut)
                try:
                    result = fut.result()
                    results.append(result)
                    log(
                        verbosity,
                        f"Fold {result['fold']} finished on GPU {gpu_id} "
                        f"(test C-index={result['test_c_index']:.4f})",
                        min_level=0,
                    )
                except Exception as exc:
                    log(verbosity, f"Fold on GPU {gpu_id} failed: {exc}", min_level=0)
                    raise
                free_gpus.append(gpu_id)
            _submit_available()

    results.sort(key=lambda r: r["fold"])
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MOGHET multi-GPU CV trainer (1 fold per GPU, up to 4 GPUs)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="BRCA",
        choices=[
            "BRCA", "LUAD", "COAD", "GBM", "KIRC", "LIHC", "LUNG", "OV", "SKCM",
            "LUSC", "STAD", "UCEC", "HNSC", "PAAD", "LGG", "PRAD",
        ],
        help="TCGA dataset name",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=200, help="Max epochs per fold")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Max concurrent GPUs / folds (default 4 for one Arrhenius node)",
    )
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=0,
        help="DataLoader num_workers per fold process (keep low; default 0)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick run: fewer patients, 2 folds, 2 epochs",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="multi_gpu",
        help="Results dir suffix: results/<dataset>_<suffix>",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="normal",
        help="quiet|normal|verbose|debug (or 0|1|2|3). Default: normal",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted run: reuse cv_splits.json, skip done folds, "
        "continue from checkpoint.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbosity = parse_verbosity(args.verbosity)
    set_seed(args.seed)

    log(verbosity, "=" * 70, min_level=0)
    log(verbosity, "  MOGHET Multi-GPU Training", min_level=0)
    log(verbosity, "=" * 70, min_level=0)
    log(verbosity, f"Dataset: {args.dataset}", min_level=0)
    log(verbosity, f"Folds: {2 if args.smoke_test else args.n_folds}", min_level=0)
    log(verbosity, f"Max concurrent GPUs: {args.num_gpus}", min_level=0)
    log(verbosity, f"Visible CUDA devices: {_available_cuda_count()}", min_level=0)
    log(verbosity, f"Verbosity: {args.verbosity} ({verbosity})", min_level=0)
    log(verbosity, f"Resume: {args.resume}", min_level=0)
    if args.smoke_test:
        log(verbosity, "Smoke-test mode: ON", min_level=0)
    log(verbosity, "=" * 70, min_level=0)

    config = build_config(
        _PROJECT_ROOT,
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

    all_patient_ids, _, _ = load_aligned_patient_ids(config)
    log(verbosity, f"Valid patients: {len(all_patient_ids)}", min_level=0)

    schedule_path = osp.join(config["output_dir"], "cv_splits.json")
    fold_splits = None
    if args.resume:
        fold_splits = load_cv_splits(config["output_dir"])
        if fold_splits is not None:
            log(
                verbosity,
                f"Loaded existing CV splits from {schedule_path} ({len(fold_splits)} folds)",
                min_level=0,
            )
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
        log(verbosity, f"Wrote CV splits to {schedule_path}", min_level=0)
    else:
        # Ensure per-fold split_info.json files exist
        save_fold_splits(config["output_dir"], fold_splits)

    t0 = time.time()
    results = run_fold_pool(
        fold_splits=fold_splits,
        config=config,
        num_gpus=args.num_gpus,
        num_workers=args.dataloader_workers,
        resume=args.resume,
        verbosity=verbosity,
    )
    elapsed = time.time() - t0

    summary = aggregate_cv_results(
        output_dir=config["output_dir"],
        n_splits=config["n_splits"],
        dataset=args.dataset,
        seed=args.seed,
        config=config,
    )

    cv = summary["cross_validation_summary"]
    log(verbosity, "=" * 70, min_level=0)
    log(verbosity, "  Multi-GPU CV finished", min_level=0)
    log(verbosity, "=" * 70, min_level=0)
    log(verbosity, f"Wall time: {elapsed / 60.0:.1f} min", min_level=0)
    log(
        verbosity,
        f"Mean test C-index: {cv['mean_test_c_index']:.4f} ± {cv['std_test_c_index']:.4f}",
        min_level=0,
    )
    log(
        verbosity,
        f"Mean test AUC:     {cv['mean_test_auc']:.4f} ± {cv['std_test_auc']:.4f}",
        min_level=0,
    )
    log(verbosity, f"Results: {config['output_dir']}", min_level=0)
    for r in results:
        log(
            verbosity,
            f"  Fold {r['fold']}: C-index={r['test_c_index']:.4f} "
            f"AUC={r['test_auc']:.4f} device={r.get('device', '?')}",
            min_level=0,
        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
