from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def recall_at_k(scores: np.ndarray, labels: np.ndarray, k: float | int) -> float:
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")

    if isinstance(k, float):
        if not 0 < k <= 1:
            raise ValueError("fractional k must be in (0,1]")
        cutoff = max(int(np.ceil(len(scores) * k)), 1)
    else:
        cutoff = max(int(k), 1)

    order = np.argsort(scores)[::-1]
    top_mask = np.zeros_like(scores, dtype=bool)
    top_mask[order[:cutoff]] = True

    true_positives = (labels.astype(bool) & top_mask).sum()
    positives = labels.astype(bool).sum()

    if positives == 0:
        return 0.0
    return float(true_positives / positives)


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: float | int) -> float:
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")

    if isinstance(k, float):
        if not 0 < k <= 1:
            raise ValueError("fractional k must be in (0,1]")
        cutoff = max(int(np.ceil(len(scores) * k)), 1)
    else:
        cutoff = max(int(k), 1)

    order = np.argsort(scores)[::-1]
    selected = labels[order[:cutoff]]

    return float(selected.mean()) if cutoff > 0 else 0.0


def average_lead_time(pred_idx: np.ndarray, true_idx: np.ndarray) -> float:
    if pred_idx.size == 0 or true_idx.size == 0:
        return 0.0

    pred_idx = np.sort(pred_idx)
    true_idx = np.sort(true_idx)

    lead_times = []
    j = 0
    for p in pred_idx:
        while j < len(true_idx) and true_idx[j] < p:
            j += 1
        if j < len(true_idx):
            lead_times.append(true_idx[j] - p)

    if not lead_times:
        return 0.0
    return float(np.mean(lead_times))


def regression_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> Dict[str, float]:
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    if pred.size == 0:
        return {"mae": 0.0, "rmse": 0.0}
    mae = float(np.mean(np.abs(pred - target)))
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    return {"mae": mae, "rmse": rmse}


def evaluate_alerts(scores: np.ndarray, labels: np.ndarray, k_fraction: float = 0.2) -> Dict[str, float]:
    return {
        "recall@top": recall_at_k(scores, labels, k_fraction),
        "precision@top": precision_at_k(scores, labels, k_fraction),
    }


def compute_lead_time_stats(pred_scores: np.ndarray, labels: np.ndarray, horizon: int) -> Dict[str, float]:
    order = np.argsort(pred_scores)[::-1]
    top_k = order[: max(int(len(order) * 0.2), 1)]
    true_indices = np.where(labels.astype(bool))[0]
    avg_lead = average_lead_time(top_k, true_indices)
    coverage = float((pred_scores[top_k] >= 0).sum() / max(len(pred_scores), 1))

    return {
        "avg_lead_time": avg_lead,
        "coverage@top": coverage,
        "horizon": float(horizon),
    }


def evaluate_multi_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    fractions: Iterable[float],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for frac in fractions:
        frac = float(frac)
        percent = int(round(frac * 100))
        metrics[f"recall@top{percent}"] = recall_at_k(scores, labels, frac)
        metrics[f"precision@top{percent}"] = precision_at_k(scores, labels, frac)
    return metrics


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    return float(np.trapz(y_sorted, x_sorted))


def precision_recall_curve(scores: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order].astype(np.float32)
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1.0 - sorted_labels)
    denom = tp + fp
    precision = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    positives = max(sorted_labels.sum(), 1.0)
    recall = tp / positives

    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])

    auc_value = _auc(recall, precision)
    return {"precision": precision, "recall": recall, "auc": auc_value}


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order].astype(np.float32)
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1.0 - sorted_labels)
    positives = max(sorted_labels.sum(), 1.0)
    negatives = max(float(len(sorted_labels) - positives), 1.0)
    tpr = tp / positives
    fpr = fp / negatives

    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])

    auc_value = _auc(fpr, tpr)
    return {"tpr": tpr, "fpr": fpr, "auc": auc_value}
