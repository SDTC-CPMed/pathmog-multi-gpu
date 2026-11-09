import sys
import os.path as osp

import numpy as np
import torch
from tqdm import tqdm

current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from evaluation.auc_evaluation import SurvivalAUCEvaluator
from training.verbosity import log


def train_one_epoch(model, loader, optimizer, device, accumulation_steps=1, verbosity: int = 1):
    """
    Run one training epoch.

    verbosity
        0 quiet, 1 normal, 2 verbose (tqdm), 3 debug (batch diagnostics).
    """
    model.train()
    total_loss = 0
    valid_batches = 0
    optimizer.zero_grad()

    use_tqdm = verbosity >= 2
    iterator = tqdm(loader, desc="Training", leave=False) if use_tqdm else loader

    for i, batch_data in enumerate(iterator):
        if batch_data is None:
            continue

        graphs_batch = batch_data["graphs_batch"].to(device)
        clinical_features = batch_data["clinical_features"].to(device)
        time = batch_data["time"].to(device)
        event = batch_data["event"].to(device)

        if torch.isnan(clinical_features).any() or torch.isinf(clinical_features).any():
            log(verbosity, f"Warning: Batch {i} clinical_features has NaN/Inf", min_level=2)
            clinical_features = torch.nan_to_num(
                clinical_features, nan=0.0, posinf=1e6, neginf=-1e6
            )

        if torch.isnan(time).any() or torch.isinf(time).any():
            log(verbosity, f"Warning: Batch {i} time has NaN/Inf", min_level=2)
            time = torch.nan_to_num(time, nan=1.0, posinf=1e6, neginf=1.0)

        if torch.isnan(event).any() or torch.isinf(event).any():
            log(verbosity, f"Warning: Batch {i} event has NaN/Inf", min_level=2)
            event = torch.nan_to_num(event, nan=0.0, posinf=1.0, neginf=0.0)

        risk_scores = model(graphs_batch, clinical_features)

        if torch.isnan(risk_scores).any():
            log(verbosity, f"Warning: NaN risk_scores in batch {i}; skipping", min_level=2)
            continue

        from .loss import cox_loss

        loss = cox_loss(risk_scores, time, event)

        if torch.isnan(loss) or torch.isinf(loss):
            log(verbosity, f"Warning: Batch {i} loss is NaN/Inf", min_level=2)
            continue

        # Cox loss can be 0 when a mini-batch has no usable events — skip quietly
        if loss.item() <= 0:
            log(verbosity, f"Warning: Batch {i} loss is non-positive: {loss.item()}", min_level=3)
            continue

        if loss.item() > 0 and not torch.isnan(loss) and not torch.isinf(loss):
            loss = loss / accumulation_steps
            loss.backward()

            total_norm = 0.0
            param_count = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        log(verbosity, "Warning: NaN/Inf gradients zeroed", min_level=2)
                        param.grad.data.zero_()

            if param_count > 0:
                total_norm = total_norm ** (1.0 / 2)
                max_grad_norm = 1.0
                if total_norm > max_grad_norm:
                    log(
                        verbosity,
                        f"Batch {i}: Gradient clipped, norm={total_norm:.4f}",
                        min_level=3,
                    )
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                if i % 10 == 0:
                    log(
                        verbosity,
                        f"Batch {i}: Loss={loss.item():.4f}, Grad={total_norm:.4f}",
                        min_level=3,
                    )

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            valid_gradients = True
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        log(verbosity, "Warning: invalid grads before step; skip", min_level=2)
                        valid_gradients = False
                        break

            if valid_gradients:
                optimizer.step()
                optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                continue

        total_loss += loss.item()
        valid_batches += 1

        if use_tqdm:
            iterator.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss / max(valid_batches, 1):.4f}",
                }
            )

    avg_loss = total_loss / max(valid_batches, 1)
    log(
        verbosity,
        f"Training completed: Valid batches={valid_batches}, Average loss={avg_loss:.4f}",
        min_level=2,
    )
    return avg_loss


def evaluate(model, loader, device, return_predictions=False, verbosity: int = 1):
    """
    Evaluate model performance (C-Index / AUC).

    verbosity controls tqdm and diagnostic prints (same scale as train_one_epoch).
    """
    model.eval()
    all_risk_scores, all_times, all_events = [], [], []
    all_patient_ids = []

    use_tqdm = verbosity >= 2
    iterator = tqdm(loader, desc="Evaluating", leave=False) if use_tqdm else loader

    with torch.no_grad():
        for batch_data in iterator:
            if batch_data is None:
                continue

            graphs_batch = batch_data["graphs_batch"].to(device)
            clinical_features = batch_data["clinical_features"].to(device)
            time = batch_data["time"]
            event = batch_data["event"]

            if torch.isnan(clinical_features).any() or torch.isinf(clinical_features).any():
                log(verbosity, "Warning: clinical_features NaN/Inf in eval", min_level=2)
                clinical_features = torch.nan_to_num(
                    clinical_features, nan=0.0, posinf=1e6, neginf=-1e6
                )

            if torch.isnan(time).any() or torch.isinf(time).any():
                log(verbosity, "Warning: time NaN/Inf in eval", min_level=2)
                time = torch.nan_to_num(time, nan=1.0, posinf=1e6, neginf=1.0)

            if torch.isnan(event).any() or torch.isinf(event).any():
                log(verbosity, "Warning: event NaN/Inf in eval", min_level=2)
                event = torch.nan_to_num(event, nan=0.0, posinf=1.0, neginf=0.0)

            risk_scores = model(graphs_batch, clinical_features)

            if torch.isnan(risk_scores).any():
                log(verbosity, "Warning: NaN risk_scores in eval; replaced with 0", min_level=2)
                risk_scores = torch.nan_to_num(risk_scores, nan=0.0)

            if torch.isinf(risk_scores).any():
                log(verbosity, "Warning: Inf risk_scores in eval; clamped", min_level=2)
                risk_scores = torch.nan_to_num(risk_scores, posinf=10.0, neginf=-10.0)

            risk_scores = torch.clamp(risk_scores, min=-10.0, max=10.0)

            all_risk_scores.append(risk_scores.cpu().numpy())
            all_times.append(time.cpu().numpy())
            all_events.append(event.cpu().numpy())

            if return_predictions and "patient_ids" in batch_data:
                all_patient_ids.extend(batch_data["patient_ids"])

    if not all_risk_scores:
        log(
            verbosity,
            "Warning: no valid predictions during evaluation; returning defaults",
            min_level=0,
        )
        return {"c_index": 0, "auc": 0.5}

    all_risk_scores = np.concatenate(all_risk_scores).squeeze()
    all_times = np.concatenate(all_times).squeeze()
    all_events = np.concatenate(all_events).squeeze()

    log(
        verbosity,
        "Evaluation data statistics:\n"
        f"  Risk scores: shape={all_risk_scores.shape}, "
        f"range=[{all_risk_scores.min():.4f}, {all_risk_scores.max():.4f}]\n"
        f"  Survival times: shape={all_times.shape}, "
        f"range=[{all_times.min():.4f}, {all_times.max():.4f}]\n"
        f"  Event status: events={np.sum(all_events)}, "
        f"censored={len(all_events) - np.sum(all_events)}",
        min_level=2,
    )

    try:
        evaluator = SurvivalAUCEvaluator()
        results = evaluator.evaluate_model_performance(
            {
                "risk_scores": all_risk_scores,
                "times": all_times,
                "events": all_events,
            }
        )

        log(
            verbosity,
            "  Evaluation metrics:\n"
            f"    C-Index: {results.get('c_index', 0.5):.4f}\n"
            f"    Time-Dependent AUC: {results.get('time_dependent_auc', 0.5):.4f}",
            min_level=2,
        )

        if "time_dependent_auc" in results and "auc" not in results:
            results["auc"] = results["time_dependent_auc"]

        if return_predictions:
            results["predictions"] = {
                "patient_ids": all_patient_ids,
                "risk_scores": all_risk_scores,
                "survival_times": all_times,
                "events": all_events,
            }

        return results
    except Exception as e:
        log(verbosity, f"  Evaluation metrics calculation failed: {e}", min_level=0)
        return {
            "c_index": 0.5,
            "auc": 0.5,
            "time_dependent_auc": 0.5,
            "integrated_auc": 0.5,
            "standard_auc": 0.5,
            "time_specific_auc": {},
            "num_samples": len(all_risk_scores),
            "num_events": int(np.sum(all_events)),
        }
