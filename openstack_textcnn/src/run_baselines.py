"""Run classic baselines and save results in the TextCNN artifact format.

This script mirrors `train.py`'s data pipeline:
- load_dataset -> split_dataset (same seed)
- train baselines on train split
- pick a threshold using val split (optional)
- evaluate on test split

Outputs
-------
{output_dir}/{run_name}/
  - metrics.json
  - confusion_matrix.png
  - test_predictions.npz

Example
-------
python -m openstack_textcnn.src.run_baselines --run-name baselines_quick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .baselines import (
    build_tfidf_linearsvm_pipeline,
    build_tfidf_logreg_pipeline,
    fit_predict_tfidf_model,
    keyword_frequency_scores,
    keyword_rule_scores,
)
from .data import ABNORMAL_KEYWORDS, load_dataset, split_dataset
from .reporting import save_confusion_matrix


def compute_metrics(
    probs: np.ndarray, labels: np.ndarray, threshold: float
) -> Tuple[Dict[str, float], np.ndarray]:
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(labels, preds).astype(int).tolist()
    auc_value = None if np.isnan(auc) else float(auc)
    metrics = {
        "loss": float("nan"),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc_value,
        "confusion_matrix": cm,
        "threshold": float(threshold),
    }
    return metrics, preds


def find_best_threshold(
    probs: np.ndarray, labels: np.ndarray, base_threshold: float
) -> Tuple[float, Dict[str, float]]:
    grid = np.linspace(0.05, 0.95, num=19)
    candidate_thresholds = np.unique(np.concatenate(([base_threshold], grid)))

    best_threshold = float(base_threshold)
    best_metrics, _ = compute_metrics(probs, labels, best_threshold)

    for threshold in candidate_thresholds:
        metrics, _ = compute_metrics(probs, labels, float(threshold))
        if metrics["f1"] > best_metrics["f1"]:
            best_metrics = metrics
            best_threshold = float(threshold)

    return best_threshold, best_metrics


def _as_int_dict(series) -> Dict[int, int]:
    return {int(k): int(v) for k, v in series.value_counts().to_dict().items()}


def _save_predictions_npz(
    output_dir: Path,
    *,
    probs: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> None:
    np.savez(
        output_dir / "test_predictions.npz",
        probs=np.asarray(probs, dtype=np.float64),
        preds=np.asarray(preds, dtype=int),
        labels=np.asarray(labels, dtype=int),
        threshold=float(threshold),
    )


def run_baseline(
    *,
    name: str,
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    val_labels: np.ndarray,
    test_texts: List[str],
    test_labels: np.ndarray,
    threshold_mode: str,
    base_threshold: float,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, float]:
    """Return (test_metrics, test_probs, test_preds, threshold)."""

    if name == "keyword_rule":
        val_probs = keyword_rule_scores(val_texts, ABNORMAL_KEYWORDS)
        test_probs = keyword_rule_scores(test_texts, ABNORMAL_KEYWORDS)
        if threshold_mode == "optimize":
            best_th, _ = find_best_threshold(
                val_probs,
                val_labels,
                base_threshold,
            )
        else:
            best_th = float(base_threshold)
        test_metrics, test_preds = compute_metrics(
            test_probs,
            test_labels,
            best_th,
        )
        return test_metrics, test_probs, test_preds, best_th

    if name == "keyword_freq_ratio":
        # score in [0,1] approx
        val_scores = keyword_frequency_scores(
            val_texts, ABNORMAL_KEYWORDS, mode="ratio"
        )
        test_scores = keyword_frequency_scores(
            test_texts, ABNORMAL_KEYWORDS, mode="ratio"
        )
        # treat score as probability
        val_probs = val_scores
        test_probs = test_scores
        if threshold_mode == "optimize":
            best_th, _ = find_best_threshold(
                val_probs,
                val_labels,
                base_threshold,
            )
        else:
            best_th = float(base_threshold)
        test_metrics, test_preds = compute_metrics(
            test_probs,
            test_labels,
            best_th,
        )
        return test_metrics, test_probs, test_preds, best_th

    if name == "tfidf_svm":
        model = build_tfidf_linearsvm_pipeline()
        # threshold is on probability after sigmoid(decision_function)
        val_res = fit_predict_tfidf_model(
            model,
            train_texts,
            train_labels,
            val_texts,
            threshold=base_threshold,
        )
        if threshold_mode == "optimize":
            best_th, _ = find_best_threshold(
                val_res.probs,
                val_labels,
                base_threshold,
            )
        else:
            best_th = float(base_threshold)
        test_res = fit_predict_tfidf_model(
            model, train_texts, train_labels, test_texts, threshold=best_th
        )
        test_metrics, _ = compute_metrics(test_res.probs, test_labels, best_th)
        return test_metrics, test_res.probs, test_res.preds, best_th

    if name == "tfidf_logreg":
        model = build_tfidf_logreg_pipeline()
        val_res = fit_predict_tfidf_model(
            model,
            train_texts,
            train_labels,
            val_texts,
            threshold=base_threshold,
        )
        if threshold_mode == "optimize":
            best_th, _ = find_best_threshold(
                val_res.probs,
                val_labels,
                base_threshold,
            )
        else:
            best_th = float(base_threshold)
        test_res = fit_predict_tfidf_model(
            model, train_texts, train_labels, test_texts, threshold=best_th
        )
        test_metrics, _ = compute_metrics(test_res.probs, test_labels, best_th)
        return test_metrics, test_res.probs, test_res.preds, best_th

    raise ValueError(f"Unknown baseline: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline models for OpenStack log anomaly detection"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "OpenStackData"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("artifacts")),
    )
    parser.add_argument("--run-name", type=str, default="baselines")

    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--normal-downsample", type=float, default=0.5)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--window-stride", type=int, default=4)

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--threshold-mode",
        choices=["fixed", "optimize"],
        default="optimize",
    )
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument(
        "--baselines",
        nargs="+",
        default=[
            "keyword_rule",
            "keyword_freq_ratio",
            "tfidf_svm",
            "tfidf_logreg",
        ],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(
        Path(args.data_dir),
        max_lines_per_file=args.max_lines,
        normal_downsample=args.normal_downsample,
        deduplicate=args.deduplicate,
        window_size=args.window_size,
        window_stride=args.window_stride,
    )
    splits = split_dataset(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.seed,
    )

    dataset_summary = {
        "total_records": int(len(df)),
        "train_size": int(len(splits.train)),
        "val_size": int(len(splits.val)),
        "test_size": int(len(splits.test)),
        "class_distribution": {
            "train": _as_int_dict(splits.train["label"]),
            "val": _as_int_dict(splits.val["label"]),
            "test": _as_int_dict(splits.test["label"]),
        },
    }

    train_texts = splits.train["text"].tolist()
    val_texts = splits.val["text"].tolist()
    test_texts = splits.test["text"].tolist()

    train_labels = splits.train["label"].to_numpy(dtype=int)
    val_labels = splits.val["label"].to_numpy(dtype=int)
    test_labels = splits.test["label"].to_numpy(dtype=int)

    all_results: Dict[str, Dict[str, float]] = {}

    for baseline_name in args.baselines:
        test_metrics, test_probs, test_preds, used_th = run_baseline(
            name=baseline_name,
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            test_texts=test_texts,
            test_labels=test_labels,
            threshold_mode=args.threshold_mode,
            base_threshold=float(args.threshold),
        )

        # Per-baseline subfolder for clear artifacts
        bdir = output_dir / baseline_name
        bdir.mkdir(parents=True, exist_ok=True)

        raw_cm = test_metrics.get("confusion_matrix")
        if (
            isinstance(raw_cm, list)
            and len(raw_cm) == 2
            and all(isinstance(r, list) and len(r) == 2 for r in raw_cm)
        ):
            cm = [[int(x) for x in r] for r in raw_cm]
        else:
            cm = [[0, 0], [0, 0]]

        save_confusion_matrix(
            cm,
            labels=["normal", "anomaly"],
            output_path=bdir / "confusion_matrix.png",
        )
        _save_predictions_npz(
            bdir,
            probs=test_probs,
            preds=test_preds,
            labels=test_labels,
            threshold=used_th,
        )

        metrics_blob = {
            "dataset": dataset_summary,
            "baseline": baseline_name,
            "decision_threshold": float(used_th),
            "test": test_metrics,
        }
        with (bdir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics_blob, f, indent=2)

        all_results[baseline_name] = test_metrics

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
