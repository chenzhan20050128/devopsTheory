from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class ForecastConfig:
    data_dir: Path = Path(__file__).resolve().parents[2] / "OpenStackData"
    output_dir: Path = Path(__file__).resolve().parents[1] / "artifacts_forecast"

    window_size: int = 8  # Match baseline
    window_stride: int = 4  # Match baseline
    sequence_length: int = 12  # Reduced for better data efficiency
    alert_horizon: int = 3  # Reduced for more focused prediction
    lead_time_minutes: Optional[int] = None  # optional real-time mapping
    lead_weight_scale: float = 1.5
    eval_k_fractions: Tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.3)

    max_lines_per_file: Optional[int] = None
    normal_downsample: float = 1.0
    deduplicate: bool = False

    batch_size: int = 32
    num_workers: int = 0
    seed: int = 42

    embedding_dim: int = 128
    numeric_dim: int = 16
    id_embedding_dim: int = 32
    time_embedding_dim: int = 16
    max_tokens: int = 200

    hidden_dim: int = 160
    num_heads: int = 4
    dropout: float = 0.2
    lambda_reg: float = 0.5
    trend_window: int = 5
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    clip_grad_norm: Optional[float] = 1.0

    learning_rate: float = 5e-4  # Reduced for more stable training
    weight_decay: float = 1e-4
    epochs: int = 20  # Increased for better convergence

    device: str = "cuda"
    use_amp: bool = True

    log_interval: int = 100
    checkpoint_path: Optional[Path] = None

    def resolve_paths(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(self.checkpoint_path)
