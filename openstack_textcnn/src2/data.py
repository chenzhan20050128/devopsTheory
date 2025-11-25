from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from src.data import Vocabulary, normalize_message as base_normalize_message
except ImportError:  # pragma: no cover
    from ..src.data import Vocabulary, normalize_message as base_normalize_message

from .config import ForecastConfig

UUID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
REQUEST_PATTERN = re.compile(r"req-[0-9a-f-]+", re.IGNORECASE)

ABNORMAL_KEYWORDS = ("error", "failed", "exception", "traceback", "critical", "panic", "fatal")


def normalize_message(message: str) -> str:
    return base_normalize_message(message)


def extract_message(line_rest: str) -> str:
    if "]" in line_rest:
        message = line_rest.split("]", maxsplit=1)[-1]
    else:
        message = line_rest
    return message.strip()


def extract_numeric_features(message: str, dim: int = 9) -> np.ndarray:
    numbers = [float(match) for match in re.findall(r"\b\d+(?:\.\d+)?\b", message)]
    if numbers:
        arr = np.asarray(numbers, dtype=np.float32)
        number_stats = np.array(
            [
                float(arr.size),
                float(arr.mean()),
                float(arr.max()),
                float(arr.min()),
                float(arr.std(ddof=0)),
            ],
            dtype=np.float32,
        )
    else:
        number_stats = np.zeros(5, dtype=np.float32)

    message_length = float(len(message)) / 200.0
    token_count = float(len(message.split())) / 50.0
    upper_ratio = sum(1 for ch in message if ch.isupper()) / max(len(message), 1)
    keyword_flag = 1.0 if any(keyword in message.lower() for keyword in ABNORMAL_KEYWORDS) else 0.0

    extra_features = np.array(
        [
            message_length,
            token_count,
            upper_ratio,
            keyword_flag,
        ],
        dtype=np.float32,
    )

    features = np.concatenate([number_stats, extra_features])
    if features.size < dim:
        features = np.concatenate([features, np.zeros(dim - features.size, dtype=np.float32)])
    return features[:dim]


def load_anomaly_instance_ids(file_path: Path) -> set[str]:
    if not file_path.exists():
        return set()
    anomaly_ids: set[str] = set()
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = UUID_PATTERN.search(line)
            if match:
                anomaly_ids.add(match.group(0).lower())
    return anomaly_ids


@dataclass(frozen=True)
class ForecastSample:
    tokens: np.ndarray
    numeric: np.ndarray
    time_features: np.ndarray
    instance_id: Optional[str]
    request_id: Optional[str]
    source: str
    group_key: str
    y_class: int
    y_rtf: float
    y_mask: bool
    weight: float


class SequenceForecastDataset(Dataset):
    def __init__(
        self,
        sequences: List[ForecastSample],
        token_pad_value: int = 0,
    ) -> None:
        self.sequences = sequences
        self.token_pad_value = token_pad_value
        self.group_to_idx: Dict[str, int] = {}
        self.group_ids: List[int] = []
        for sample in self.sequences:
            key = sample.group_key or sample.instance_id or sample.request_id or sample.source or "unknown"
            if key not in self.group_to_idx:
                self.group_to_idx[key] = len(self.group_to_idx)
            self.group_ids.append(self.group_to_idx[key])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sequences[idx]
        return {
            "tokens": torch.as_tensor(sample.tokens, dtype=torch.long),
            "numeric": torch.as_tensor(sample.numeric, dtype=torch.float32),
            "time": torch.as_tensor(sample.time_features, dtype=torch.float32),
            "group_id": torch.tensor(self.group_ids[idx], dtype=torch.long),
            "y_class": torch.tensor(sample.y_class, dtype=torch.float32),
            "y_rtf": torch.tensor(sample.y_rtf, dtype=torch.float32),
            "y_mask": torch.tensor(sample.y_mask, dtype=torch.bool),
            "weight": torch.tensor(sample.weight, dtype=torch.float32),
        }

    @staticmethod
    def collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        tokens = torch.stack([item["tokens"] for item in batch], dim=0)
        numeric = torch.stack([item["numeric"] for item in batch], dim=0)
        time_feat = torch.stack([item["time"] for item in batch], dim=0)
        group_ids = torch.stack([item["group_id"] for item in batch], dim=0)
        y_class = torch.stack([item["y_class"] for item in batch], dim=0)
        y_rtf = torch.stack([item["y_rtf"] for item in batch], dim=0)
        y_mask = torch.stack([item["y_mask"] for item in batch], dim=0)
        weights = torch.stack([item["weight"] for item in batch], dim=0)
        return {
            "tokens": tokens,
            "numeric": numeric,
            "time": time_feat,
            "group_id": group_ids,
            "y_class": y_class,
            "y_rtf": y_rtf,
            "y_mask": y_mask,
            "weight": weights,
        }


def _compute_time_features(timestamps: Sequence[pd.Timestamp]) -> np.ndarray:
    hours = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps], dtype=np.float32)
    days = np.array([ts.dayofweek for ts in timestamps], dtype=np.float32)

    hours_sin = np.sin(2 * math.pi * hours / 24.0)
    hours_cos = np.cos(2 * math.pi * hours / 24.0)

    days_sin = np.sin(2 * math.pi * days / 7.0)
    days_cos = np.cos(2 * math.pi * days / 7.0)

    return np.stack([hours_sin, hours_cos, days_sin, days_cos], axis=1)


def _extract_ids(line: str) -> Tuple[Optional[str], Optional[str]]:
    inst = UUID_PATTERN.search(line)
    req = REQUEST_PATTERN.search(line)
    return (inst.group(0).lower() if inst else None, req.group(0).lower() if req else None)


def _parse_log_line(line: str) -> Tuple[Optional[str], Optional[pd.Timestamp], str]:
    parts = line.strip().split(" ", maxsplit=4)
    if len(parts) < 5:
        return None, None, line.strip()
    source = parts[0]
    timestamp_str = f"{parts[1]} {parts[2]}"
    try:
        timestamp = pd.to_datetime(timestamp_str)
    except (ValueError, TypeError):
        timestamp = None
    return source, timestamp, parts[4]


def _determine_group(instance_id: Optional[str], request_id: Optional[str], source: str) -> str:
    if instance_id:
        return f"inst:{instance_id}"
    if request_id:
        return f"req:{request_id}"
    return f"src:{source}"


def load_raw_dataframe(
    data_dir: Path,
    max_lines_per_file: Optional[int],
    normal_downsample: float,
) -> pd.DataFrame:
    files = [
        (data_dir / "openstack_normal1.log", 0),
        (data_dir / "openstack_normal2.log", 0),
        (data_dir / "openstack_abnormal.log", 1),
    ]

    anomaly_ids = load_anomaly_instance_ids(data_dir / "anomaly_labels.txt")

    records: List[Dict[str, object]] = []
    for file_path, base_label in files:
        if not file_path.exists():
            continue
        tracked_requests: set[str] = set()
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for idx, raw_line in enumerate(handle):
                if max_lines_per_file is not None and idx >= max_lines_per_file:
                    break

                source, timestamp, remainder = _parse_log_line(raw_line)
                message = extract_message(remainder)
                if not message:
                    continue

                normalized = normalize_message(message)
                if not normalized:
                    continue

                instance_id, request_id = _extract_ids(raw_line)
                numeric = extract_numeric_features(message)

                request_ids = [req.lower() for req in REQUEST_PATTERN.findall(raw_line)]
                effective_label = base_label
                if anomaly_ids:
                    if instance_id and instance_id in anomaly_ids:
                        effective_label = 1
                    elif base_label == 1 and request_ids and any(req in tracked_requests for req in request_ids):
                        effective_label = 1
                    elif base_label == 1 and any(keyword in normalized for keyword in ABNORMAL_KEYWORDS):
                        effective_label = 1
                    else:
                        effective_label = 0

                    if effective_label == 1 and request_ids:
                        tracked_requests.update(request_ids)

                if effective_label == 0 and base_label == 0 and normal_downsample < 1.0 and random.random() > normal_downsample:
                    continue

                records.append(
                    {
                        "timestamp": timestamp,
                        "text": normalized,
                        "raw_text": message,
                        "label": int(effective_label),
                        "instance_id": instance_id,
                        "request_id": request_id,
                        "source": source or file_path.name,
                        "features": numeric,
                    }
                )

    df = pd.DataFrame(records)
    df = df.dropna(subset=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["group_key"] = [
        _determine_group(inst, req, src)
        for inst, req, src in zip(df["instance_id"], df["request_id"], df["source"])
    ]
    return df


def create_windows(
    df: pd.DataFrame,
    window_size: int,
    window_stride: int,
    trend_window: int,
) -> pd.DataFrame:
    columns = [
        "timestamp",
        "text",
        "label",
        "instance_id",
        "request_id",
        "source",
        "features",
        "group_key",
    ]
    window_records: List[Tuple[object, ...]] = []

    for group_key, group in df.groupby("group_key", sort=False):
        group_sorted = group.sort_values("timestamp")
        texts = group_sorted["text"].tolist()
        timestamps = group_sorted["timestamp"].tolist()
        labels = group_sorted["label"].tolist()
        features = np.stack(group_sorted["features"].to_numpy())
        instance_ids = group_sorted["instance_id"].tolist()
        request_ids = group_sorted["request_id"].tolist()
        sources = group_sorted["source"].tolist()
        recent_ratios: List[float] = []

        for start in range(0, len(group_sorted) - window_size + 1, window_stride):
            end = start + window_size
            window_text = " ".join(texts[start:end])
            label_slice = labels[start:end]
            window_label = int(any(label_slice))
            base_features = features[start:end].mean(axis=0)

            anomaly_ratio = float(sum(label_slice)) / max(window_size, 1)

            if pd.isna(timestamps[end - 1]) or pd.isna(timestamps[start]):
                duration = 0.0
            else:
                duration = (timestamps[end - 1] - timestamps[start]).total_seconds()
            duration_feature = float(duration) / 600.0

            intervals: List[float] = []
            for idx in range(start + 1, end):
                t_prev = timestamps[idx - 1]
                t_curr = timestamps[idx]
                if pd.isna(t_prev) or pd.isna(t_curr):
                    continue
                intervals.append((t_curr - t_prev).total_seconds())
            if intervals:
                avg_interval = float(np.mean(intervals)) / 60.0
                std_interval = float(np.std(intervals)) / 60.0
            else:
                avg_interval = 0.0
                std_interval = 0.0

            spike_count = sum(
                1 for idx in range(1, len(label_slice)) if label_slice[idx - 1] == 0 and label_slice[idx] == 1
            )
            spike_feature = float(spike_count) / max(window_size - 1, 1)

            if recent_ratios:
                history_len = max(trend_window, 1)
                history = recent_ratios[-history_len:]
                history_mean = float(np.mean(history))
                history_delta = anomaly_ratio - history_mean
            else:
                history_mean = anomaly_ratio
                history_delta = 0.0

            window_features = np.concatenate(
                [
                    base_features,
                    np.array(
                        [
                            anomaly_ratio,
                            duration_feature,
                            avg_interval,
                            std_interval,
                            spike_feature,
                            history_mean,
                            history_delta,
                        ],
                        dtype=np.float32,
                    ),
                ]
            )

            window_timestamp = timestamps[end - 1]
            window_instance = next((inst for inst in reversed(instance_ids[start:end]) if inst), None)
            window_request = next((req for req in reversed(request_ids[start:end]) if req), None)
            window_source = sources[end - 1]

            window_records.append(
                (
                    window_timestamp,
                    window_text,
                    window_label,
                    window_instance,
                    window_request,
                    window_source,
                    window_features,
                    group_key,
                )
            )
            recent_ratios.append(anomaly_ratio)

    window_df = pd.DataFrame(window_records, columns=columns)
    window_df.sort_values("timestamp", inplace=True)
    window_df.reset_index(drop=True, inplace=True)
    return window_df


def pad_tokens(token_ids: List[int], max_length: int) -> np.ndarray:
    if len(token_ids) >= max_length:
        return np.array(token_ids[:max_length], dtype=np.int64)
    padded = token_ids + [0] * (max_length - len(token_ids))
    return np.array(padded, dtype=np.int64)


def encode_tokens(df: pd.DataFrame, vocab: Vocabulary, max_tokens: int) -> pd.DataFrame:
    token_arrays = [pad_tokens(vocab.encode(text), max_tokens) for text in df["text"].tolist()]
    df = df.copy()
    df["tokens"] = token_arrays
    return df


def time_based_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    total = len(df_sorted)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()
    return train_df, val_df, test_df


def _backfill_labels(
    labels: np.ndarray,
    alert_horizon: int,
    lead_weight_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate forecast labels by looking forward from each window.
    For each window at index i, check if there's an anomaly within alert_horizon windows ahead.
    """
    n = len(labels)
    y_class = np.zeros_like(labels, dtype=np.int64)
    y_rtf = np.full(n, fill_value=float(alert_horizon + 1), dtype=np.float32)
    y_mask = np.ones(n, dtype=bool)  # Changed: mask all samples for regression, not just positive ones
    weights = np.ones(n, dtype=np.float32)

    # Forward-looking approach: for each window, check future windows
    for idx in range(n):
        # Look ahead within horizon (including current window if it's anomalous)
        look_ahead_end = min(idx + alert_horizon + 1, n)
        
        # Find the first anomaly in the future (including current)
        future_anomaly_idx = None
        for j in range(idx, look_ahead_end):
            if labels[j] == 1:
                future_anomaly_idx = j
                break
        
        if future_anomaly_idx is not None:
            # Found an anomaly within horizon
            distance = future_anomaly_idx - idx
            y_class[idx] = 1
            y_rtf[idx] = float(distance)
            y_mask[idx] = True
            # Weight closer predictions more heavily
            if alert_horizon > 0:
                weight_factor = 1.0 - distance / alert_horizon
                weights[idx] = 1.0 + lead_weight_scale * max(weight_factor, 0.1)
            else:
                weights[idx] = 1.0 + lead_weight_scale
        else:
            # No anomaly in horizon, but still valid for training
            # Set RTF to horizon + 1 to indicate "far future"
            y_rtf[idx] = float(alert_horizon + 1)
            y_mask[idx] = True  # Still compute loss for negative samples
            weights[idx] = 1.0

    return y_class, y_rtf, y_mask, weights


def build_sequences(
    df: pd.DataFrame,
    token_col: str,
    numeric_col: str,
    timestamp_col: str,
    sequence_length: int,
    alert_horizon: int,
    lead_weight_scale: float,
    group_col: str = "instance_id",
) -> List[ForecastSample]:
    sequences: List[ForecastSample] = []

    for group_key, group in df.groupby(group_col):
        # Sort by timestamp to ensure temporal order
        group = group.sort_values(timestamp_col).reset_index(drop=True)
        
        if len(group) < sequence_length:
            continue

        tokens_array = np.stack(group[token_col].to_numpy())
        numeric_array = np.stack(group[numeric_col].to_numpy())
        timestamps = pd.to_datetime(group[timestamp_col].to_numpy())
        time_feats = _compute_time_features(timestamps)
        labels = group["label"].to_numpy(dtype=np.int64)

        # Generate forecast labels for the entire group
        y_class, y_rtf, y_mask, weights = _backfill_labels(labels, alert_horizon, lead_weight_scale)

        # Create sequences with adaptive stride to balance data usage and diversity
        # Use smaller stride for smaller groups to maximize data
        stride = max(1, min(sequence_length // 3, len(group) // 10))
        for start in range(0, len(group) - sequence_length + 1, stride):
            end = start + sequence_length
            tokens_slice = tokens_array[start:end]
            numeric_slice = numeric_array[start:end]
            time_slice = time_feats[start:end]

            # Use the label at the END of the sequence (what we're predicting)
            # This represents: given sequence [t-k+1:t], predict if anomaly occurs in [t+1:t+H]
            label_slice = y_class[end - 1]
            rtf_slice = y_rtf[end - 1]
            mask_slice = y_mask[end - 1]
            weight_slice = weights[end - 1]

            sequences.append(
                ForecastSample(
                    tokens=tokens_slice,
                    numeric=numeric_slice,
                    time_features=time_slice,
                    instance_id=group.iloc[end - 1].get("instance_id"),
                    request_id=group.iloc[end - 1].get("request_id"),
                    source=str(group.iloc[end - 1].get("source", "")),
                    group_key=str(group_key),
                    y_class=int(label_slice),
                    y_rtf=float(rtf_slice),
                    y_mask=bool(mask_slice),
                    weight=float(weight_slice),
                )
            )

    return sequences


def split_sequences(
    sequences: Sequence[ForecastSample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[ForecastSample], List[ForecastSample], List[ForecastSample]]:
    rng = random.Random(seed)
    seq_list = list(sequences)
    rng.shuffle(seq_list)

    total = len(seq_list)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train = seq_list[:train_end]
    val = seq_list[train_end:val_end]
    test = seq_list[val_end:]
    return train, val, test


def prepare_datasets(config: ForecastConfig) -> Tuple[SequenceForecastDataset, SequenceForecastDataset, SequenceForecastDataset, Vocabulary]:
    raw_df = load_raw_dataframe(
        data_dir=Path(config.data_dir),
        max_lines_per_file=config.max_lines_per_file,
        normal_downsample=config.normal_downsample,
    )

    window_df = create_windows(
        raw_df,
        window_size=config.window_size,
        window_stride=config.window_stride,
        trend_window=config.trend_window,
    )

    train_windows, val_windows, test_windows = time_based_split(
        window_df,
        train_ratio=0.7,
        val_ratio=0.15,
    )

    vocab = Vocabulary(min_freq=2, max_size=20000)
    vocab.build(train_windows["text"].tolist())

    train_windows = encode_tokens(train_windows, vocab, config.max_tokens)
    val_windows = encode_tokens(val_windows, vocab, config.max_tokens)
    test_windows = encode_tokens(test_windows, vocab, config.max_tokens)

    train_sequences = build_sequences(
        train_windows,
        token_col="tokens",
        numeric_col="features",
        timestamp_col="timestamp",
        sequence_length=config.sequence_length,
        alert_horizon=config.alert_horizon,
        lead_weight_scale=config.lead_weight_scale,
        group_col="group_key",
    )
    val_sequences = build_sequences(
        val_windows,
        token_col="tokens",
        numeric_col="features",
        timestamp_col="timestamp",
        sequence_length=config.sequence_length,
        alert_horizon=config.alert_horizon,
        lead_weight_scale=config.lead_weight_scale,
        group_col="group_key",
    )
    test_sequences = build_sequences(
        test_windows,
        token_col="tokens",
        numeric_col="features",
        timestamp_col="timestamp",
        sequence_length=config.sequence_length,
        alert_horizon=config.alert_horizon,
        lead_weight_scale=config.lead_weight_scale,
        group_col="group_key",
    )

    train_ds = SequenceForecastDataset(train_sequences)
    val_ds = SequenceForecastDataset(val_sequences)
    test_ds = SequenceForecastDataset(test_sequences)
    return train_ds, val_ds, test_ds, vocab
