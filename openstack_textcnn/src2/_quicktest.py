from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from src2.config import ForecastConfig
    from src2.data import prepare_datasets
else:
    from .config import ForecastConfig
    from .data import prepare_datasets

if __name__ == "__main__":
    cfg = ForecastConfig()
    cfg.max_lines_per_file = 10
    cfg.window_size = 2
    cfg.window_stride = 1
    cfg.sequence_length = 2
    cfg.alert_horizon = 1
    cfg.normal_downsample = 0.5

    train_ds, val_ds, test_ds, vocab = prepare_datasets(cfg)
    print("sizes:", len(train_ds), len(val_ds), len(test_ds))
    print("vocab:", len(vocab))
