#!/usr/bin/env python
"""
Shared single-fold training helpers used by sequential and multi-GPU trainers.

Keep fold training logic here so orchestrators stay thin (DRY).
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader

# Project root on sys.path for package imports
_CURRENT_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.data_loader import HierarchicalDataset, hierarchical_collate_fn
from models import HierarchicalGNNModel
from training.engine import evaluate, train_one_epoch
from training.verbosity import log


def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility within a process."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_jsonable(obj: Any) -> Any:
    """Convert nested tuples (e.g. PyG metadata) into JSON-serializable lists."""
    if isinstance(obj, tuple):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def build_config(
    project_root: str,
    dataset_name: str,
    *,
    seed: int = 42,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-4,
    n_folds: int = 5,
    patience: int = 20,
    smoke_test: bool = False,
    output_suffix: str = "model",
    num_pathways: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build the training config dict shared by trainers.

    Parameters
    ----------
    output_suffix : str
        Suffix after dataset name in results/<dataset>_<suffix>.
    """
    processed_data_dir = osp.join(project_root, "data", "processed", dataset_name)
    if num_pathways is None:
        with open(osp.join(processed_data_dir, "id_mappings.json"), "r") as f:
            id_mappings = json.load(f)
        num_pathways = len(id_mappings["pathway_id_to_idx"])

    suffix = f"{dataset_name.lower()}_{output_suffix}"
    if smoke_test:
        suffix = f"{dataset_name.lower()}_{output_suffix}_smoke"

    config: Dict[str, Any] = {
        "dataset": dataset_name,
        "data_path": osp.join(processed_data_dir, "hierarchical_patient_data"),
        "clinical_data_path": osp.join(processed_data_dir, "patient_clinical_features.csv"),
        "survival_data_path": osp.join(processed_data_dir, "patient_survival.csv"),
        "id_mappings_path": osp.join(processed_data_dir, "id_mappings.json"),
        "output_dir": osp.join(project_root, "results", suffix),
        "epochs": 2 if smoke_test else epochs,
        "batch_size": batch_size,
        "lr": lr,
        "n_splits": 2 if smoke_test else n_folds,
        "test_size": 0.2,
        "random_state": seed,
        "accumulation_steps": 4,
        "smoke_test": smoke_test,
        "model_params": {
            "num_pathways": num_pathways,
            "pathway_embedding_dim": 8,
            "gnn_hidden_channels": 64,
            "pathway_out_channels": 128,
            "intra_attention_hidden_channels": 128,
            "inter_attention_hidden_channels": 128,
            "clinical_hidden_channels": 32,
            "final_hidden_channels": 64,
        },
        "early_stopping": {
            "patience": patience if not smoke_test else 2,
            "min_delta": 0.001,
        },
    }
    return config


def load_aligned_patient_ids(config: Dict[str, Any]) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Load clinical/survival tables and return patient IDs that have graph files.

    Returns
    -------
    all_patient_ids, survival_df, clinical_df
    """
    survival_df = pd.read_csv(config["survival_data_path"], index_col=0)
    clinical_df = pd.read_csv(config["clinical_data_path"], index_col=0)

    common_patients = sorted(list(set(survival_df.index) & set(clinical_df.index)))
    data_files_path = config["data_path"]
    existing_patient_ids = {
        f.replace(".pt", "") for f in os.listdir(data_files_path) if f.endswith(".pt")
    }
    all_patient_ids = np.array([pid for pid in common_patients if pid in existing_patient_ids])

    if config.get("smoke_test"):
        all_patient_ids = all_patient_ids[: min(50, len(all_patient_ids))]

    return all_patient_ids, survival_df, clinical_df


def prepare_cv_splits(
    all_patient_ids: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, List[Dict[str, List[str]]]]:
    """
    Create a held-out test set and K-fold train/val splits on the remainder.

    Returns
    -------
    test_patient_ids : np.ndarray
    fold_splits : list of dicts with keys train/val/test_patient_ids (lists)
                  Length equals config['n_splits']; fold index is 0-based in the list.
    """
    train_val_ids, test_patient_ids = train_test_split(
        all_patient_ids,
        test_size=config["test_size"],
        random_state=config["random_state"],
        shuffle=True,
    )
    kf = KFold(
        n_splits=config["n_splits"],
        shuffle=True,
        random_state=config["random_state"],
    )
    fold_splits: List[Dict[str, List[str]]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_val_ids)):
        fold_splits.append(
            {
                "fold": fold_idx + 1,
                "train_patient_ids": train_val_ids[train_idx].tolist(),
                "val_patient_ids": train_val_ids[val_idx].tolist(),
                "test_patient_ids": test_patient_ids.tolist(),
            }
        )
    return test_patient_ids, fold_splits


def save_fold_splits(output_dir: str, fold_splits: Sequence[Dict[str, Any]]) -> None:
    """Write split_info.json under each fold_* directory before training starts."""
    for split in fold_splits:
        fold_dir = osp.join(output_dir, f"fold_{split['fold']}")
        os.makedirs(fold_dir, exist_ok=True)
        with open(osp.join(fold_dir, "split_info.json"), "w") as f:
            json.dump(split, f, indent=2)


def load_cv_splits(output_dir: str) -> Optional[List[Dict[str, Any]]]:
    """Load previously saved ``cv_splits.json`` if present (needed for --resume)."""
    path = osp.join(output_dir, "cv_splits.json")
    if not osp.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def fold_is_complete(output_dir: str, fold: int) -> bool:
    """True if this fold already wrote fold_result.json (finished train+test)."""
    return osp.exists(osp.join(output_dir, f"fold_{fold}", "fold_result.json"))


def load_fold_result(output_dir: str, fold: int) -> Optional[Dict[str, Any]]:
    path = osp.join(output_dir, f"fold_{fold}", "fold_result.json")
    if not osp.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def train_one_fold(
    fold: int,
    train_patient_ids: Sequence[str],
    val_patient_ids: Sequence[str],
    test_patient_ids: Sequence[str],
    config: Dict[str, Any],
    device: torch.device,
    *,
    num_workers: int = 0,
    resume: bool = False,
    verbosity: int = 1,
) -> Dict[str, Any]:
    """
    Train and evaluate a single CV fold on ``device``.

    Reloads clinical/survival CSVs from ``config`` paths so worker processes
    do not need large pickled DataFrames.

    With ``resume=True``:
    - skip entirely if ``fold_result.json`` exists
    - otherwise continue from ``checkpoint.pt`` when present
    """
    fold_output_dir = osp.join(config["output_dir"], f"fold_{fold}")
    os.makedirs(fold_output_dir, exist_ok=True)

    if resume and fold_is_complete(config["output_dir"], fold):
        result = load_fold_result(config["output_dir"], fold)
        log(
            verbosity,
            f"[Fold {fold}] Already complete — skipping "
            f"(test C-index={result.get('test_c_index', float('nan')):.4f})",
            min_level=0,
        )
        return result

    set_seed(config["random_state"] + fold)

    survival_df = pd.read_csv(config["survival_data_path"], index_col=0)
    clinical_df = pd.read_csv(config["clinical_data_path"], index_col=0)

    train_patient_ids = np.array(list(train_patient_ids))
    val_patient_ids = np.array(list(val_patient_ids))
    test_patient_ids = np.array(list(test_patient_ids))

    log(
        verbosity,
        f"[Fold {fold}] device={device} | "
        f"train={len(train_patient_ids)} val={len(val_patient_ids)} "
        f"test={len(test_patient_ids)}",
        min_level=0,
    )

    train_clinical_df = clinical_df.loc[train_patient_ids]
    val_clinical_df = clinical_df.loc[val_patient_ids]
    test_clinical_df = clinical_df.loc[test_patient_ids]

    numeric_features = train_clinical_df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = train_clinical_df.select_dtypes(include="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="passthrough",
    )

    train_features = preprocessor.fit_transform(train_clinical_df)
    val_features = preprocessor.transform(val_clinical_df)
    test_features = preprocessor.transform(test_clinical_df)

    train_features_dict = {pid: f for pid, f in zip(train_patient_ids, train_features)}
    val_features_dict = {pid: f for pid, f in zip(val_patient_ids, val_features)}
    test_features_dict = {pid: f for pid, f in zip(test_patient_ids, test_features)}

    # HierarchicalDataset prints on init; silence under quiet/normal
    _stdout = sys.stdout
    if verbosity < 2:
        sys.stdout = open(os.devnull, "w")
    try:
        train_dataset = HierarchicalDataset(
            config["data_path"], train_patient_ids, survival_df, train_features_dict
        )
        val_dataset = HierarchicalDataset(
            config["data_path"], val_patient_ids, survival_df, val_features_dict
        )
        test_dataset = HierarchicalDataset(
            config["data_path"], test_patient_ids, survival_df, test_features_dict
        )
    finally:
        if verbosity < 2:
            sys.stdout.close()
            sys.stdout = _stdout

    # Keep DataLoader workers low: graph loading is I/O-bound and high counts can hang
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=hierarchical_collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=hierarchical_collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=hierarchical_collate_fn,
        num_workers=num_workers,
    )

    sample_data = train_dataset[0]["intra_pathway_graphs"][0]
    metadata = sample_data.metadata()
    clinical_in_features = train_features.shape[1]

    fold_params = {
        "metadata": _to_jsonable(metadata),
        "clinical_in_features": int(clinical_in_features),
        **config["model_params"],
    }
    with open(osp.join(fold_output_dir, "model_params.json"), "w") as f:
        json.dump(fold_params, f, indent=2)

    split_info = {
        "fold": fold,
        "train_patient_ids": train_patient_ids.tolist(),
        "val_patient_ids": val_patient_ids.tolist(),
        "test_patient_ids": test_patient_ids.tolist(),
        "device": str(device),
    }
    with open(osp.join(fold_output_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    model = HierarchicalGNNModel(
        metadata=metadata,
        clinical_in_features=clinical_in_features,
        **config["model_params"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    start_epoch = 0
    best_val_c_index = 0.0
    best_val_auc = 0.5
    best_epoch = -1
    early_stopping_counter = 0

    checkpoint_file = osp.join(fold_output_dir, "checkpoint.pt")
    best_model_file = osp.join(fold_output_dir, "best_model.pt")

    if resume and osp.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_val_c_index = float(checkpoint["best_val_c_index"])
        best_val_auc = float(checkpoint["best_val_auc"])
        best_epoch = int(checkpoint["best_epoch"])
        early_stopping_counter = int(checkpoint["early_stopping_counter"])
        log(
            verbosity,
            f"[Fold {fold}] Resumed from epoch {start_epoch} "
            f"(best val C-index={best_val_c_index:.4f} @ epoch {best_epoch})",
            min_level=0,
        )

    for epoch in range(start_epoch, config["epochs"]):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config["accumulation_steps"],
            verbosity=verbosity,
        )
        val_results = evaluate(model, val_loader, device, verbosity=verbosity)
        val_c_index = val_results["c_index"]
        val_auc = val_results["auc"]

        log(
            verbosity,
            f"[Fold {fold}] Epoch {epoch + 1:03d}/{config['epochs']:03d} | "
            f"Loss: {train_loss:.4f} | Val C-Index: {val_c_index:.4f} | Val AUC: {val_auc:.4f}",
            min_level=1,
        )

        if val_c_index > best_val_c_index + config["early_stopping"]["min_delta"]:
            best_val_c_index = val_c_index
            best_val_auc = val_auc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_file)
            log(
                verbosity,
                f"[Fold {fold}] Saved best model "
                f"(Epoch {best_epoch}, C-Index: {best_val_c_index:.4f})",
                min_level=1,
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= config["early_stopping"]["patience"]:
                log(
                    verbosity,
                    f"[Fold {fold}] Early stopping at epoch {epoch + 1}",
                    min_level=0,
                )
                break

        # Persist after every epoch so a killed session can resume
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_c_index": best_val_c_index,
                "best_val_auc": best_val_auc,
                "best_epoch": best_epoch,
                "early_stopping_counter": early_stopping_counter,
            },
            checkpoint_file,
        )

    if not osp.exists(best_model_file):
        # Never improved: fall back to last checkpoint weights
        log(
            verbosity,
            f"[Fold {fold}] No best_model.pt; using last checkpoint for test",
            min_level=0,
        )
        torch.save(model.state_dict(), best_model_file)

    model.load_state_dict(torch.load(best_model_file, map_location=device, weights_only=False))
    test_results = evaluate(
        model, test_loader, device, return_predictions=True, verbosity=verbosity
    )
    test_c_index = test_results["c_index"]
    test_auc = test_results["auc"]

    if "predictions" in test_results:
        predictions_df = pd.DataFrame(
            {
                "patient_id": test_patient_ids,
                "risk_score": test_results["predictions"]["risk_scores"],
                "survival_time": test_results["predictions"]["survival_times"],
                "event": test_results["predictions"]["events"],
            }
        )
        predictions_df.to_csv(osp.join(fold_output_dir, "test_predictions.csv"), index=False)

    result = {
        "fold": fold,
        "device": str(device),
        "best_validation_c_index": float(best_val_c_index),
        "best_validation_auc": float(best_val_auc),
        "test_c_index": float(test_c_index),
        "test_auc": float(test_auc),
        "best_epoch": int(best_epoch),
    }
    with open(osp.join(fold_output_dir, "fold_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    log(
        verbosity,
        f"[Fold {fold}] Done | Test C-Index: {test_c_index:.4f} | AUC: {test_auc:.4f}",
        min_level=0,
    )
    return result


def aggregate_cv_results(
    output_dir: str,
    n_splits: int,
    dataset: str,
    seed: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Collect per-fold results and write summary + aggregated predictions."""
    fold_results: List[Dict[str, Any]] = []
    test_c_indices: List[float] = []
    test_aucs: List[float] = []

    for fold in range(1, n_splits + 1):
        result_path = osp.join(output_dir, f"fold_{fold}", "fold_result.json")
        if not osp.exists(result_path):
            print(f"Warning: missing {result_path}")
            continue
        with open(result_path, "r") as f:
            result = json.load(f)
        fold_results.append(result)
        test_c_indices.append(result["test_c_index"])
        test_aucs.append(result["test_auc"])

    mean_c = float(np.mean(test_c_indices)) if test_c_indices else float("nan")
    std_c = float(np.std(test_c_indices)) if test_c_indices else float("nan")
    mean_auc = float(np.mean(test_aucs)) if test_aucs else float("nan")
    std_auc = float(np.std(test_aucs)) if test_aucs else float("nan")

    summary = {
        "dataset": dataset,
        "seed": seed,
        "cross_validation_summary": {
            "mean_test_c_index": mean_c,
            "std_test_c_index": std_c,
            "all_test_c_indices": test_c_indices,
            "mean_test_auc": mean_auc,
            "std_test_auc": std_auc,
            "all_test_aucs": test_aucs,
        },
        "fold_details": fold_results,
        "config": config,
    }
    with open(osp.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    all_predictions = []
    for fold in range(1, n_splits + 1):
        pred_path = osp.join(output_dir, f"fold_{fold}", "test_predictions.csv")
        if osp.exists(pred_path):
            fold_preds = pd.read_csv(pred_path)
            fold_preds["fold"] = fold
            all_predictions.append(fold_preds)

    if all_predictions:
        combined = pd.concat(all_predictions, ignore_index=True)
        combined.to_csv(osp.join(output_dir, "all_test_predictions.csv"), index=False)
        avg = (
            combined.groupby("patient_id")
            .agg(
                {
                    "risk_score": "mean",
                    "survival_time": "first",
                    "event": "first",
                    "fold": lambda x: ",".join(map(str, sorted(x))),
                }
            )
            .reset_index()
        )
        avg.to_csv(osp.join(output_dir, "averaged_test_predictions.csv"), index=False)
        avg[["patient_id", "risk_score"]].to_csv(
            osp.join(output_dir, "all_patients_predictions.csv"), index=False
        )

    return summary
