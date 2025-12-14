"""Compare TextCNN results with baseline results and export a table.

This is a small utility to generate a thesis-friendly comparison table from
existing artifacts.

- Reads TextCNN metrics.json produced by `train.py`
- Reads baseline metrics.json produced by `run_baselines.py`
- Exports:
  - comparison_table.md
  - comparison_table.csv

Example
-------
python -m openstack_textcnn.src.compare_runs \
  --textcnn-metrics openstack_textcnn/artifacts/latest_run/metrics.json \
  --baseline-root artifacts/baselines_smoke \
  --out artifacts/baselines_smoke/comparison
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


METRIC_KEYS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "-"


def _extract_test_metrics(blob: Dict[str, Any]) -> Dict[str, Optional[float]]:
    test = blob.get("test", {}) if isinstance(blob, dict) else {}
    out: Dict[str, Optional[float]] = {}
    for k in METRIC_KEYS:
        v = test.get(k)
        out[k] = None if v is None else float(v)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare TextCNN and baseline runs",
    )
    p.add_argument(
        "--textcnn-metrics",
        type=str,
        default=str(
            Path("openstack_textcnn") / "artifacts" / "latest_run" / "metrics.json"
        ),
    )
    p.add_argument(
        "--baseline-root",
        type=str,
        default=str(Path("artifacts") / "baselines_smoke"),
        help="Should contain subfolders like tfidf_svm/metrics.json",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(Path("artifacts") / "comparison"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []

    textcnn_blob = _load_json(Path(args.textcnn_metrics))
    textcnn_metrics = _extract_test_metrics(textcnn_blob)
    rows.append(
        {
            "model": "TextCNN",
            **{k: _fmt(textcnn_metrics[k]) for k in METRIC_KEYS},
        }
    )

    baseline_root = Path(args.baseline_root)
    for sub in sorted([p for p in baseline_root.iterdir() if p.is_dir()]):
        metrics_path = sub / "metrics.json"
        if not metrics_path.exists():
            continue
        blob = _load_json(metrics_path)
        m = _extract_test_metrics(blob)
        rows.append(
            {
                "model": sub.name,
                **{k: _fmt(m[k]) for k in METRIC_KEYS},
            }
        )

    # Markdown
    md_lines = []
    md_lines.append("| 模型 | Acc | Precision | Recall | F1 | AUC |")
    md_lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        line_tpl = "| {model} | {accuracy} | {precision} | {recall} | {f1} | {auc} |"
        md_lines.append(line_tpl.format(**r))

    (out_dir / "comparison_table.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )

    # CSV
    with (out_dir / "comparison_table.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=["model", *METRIC_KEYS])
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "model": r["model"],
                    **{k: r[k] for k in METRIC_KEYS},
                }
            )

    print((out_dir / "comparison_table.md").as_posix())


if __name__ == "__main__":
    main()
