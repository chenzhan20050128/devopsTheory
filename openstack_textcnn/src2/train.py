from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ForecastConfig
from .data import SequenceForecastDataset, prepare_datasets
from .metrics import (
    compute_lead_time_stats,
    evaluate_alerts,
    regression_metrics,
    evaluate_multi_thresholds,
    precision_recall_curve,
    roc_curve,
)
from .model import ForecastModel


def setup_logging(output_dir: Path | None = None) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger("forecast_training")
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果提供了输出目录）
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_dir / "training.log", encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weight(dataset: SequenceForecastDataset) -> float:
    labels = np.array([sample.y_class for sample in dataset.sequences], dtype=np.float32)
    positives = labels.sum()
    negatives = len(labels) - positives
    positives = max(positives, 1.0)
    negatives = max(negatives, 1.0)
    return float(negatives / positives)


def build_dataloaders(
    train_ds: SequenceForecastDataset,
    val_ds: SequenceForecastDataset,
    test_ds: SequenceForecastDataset,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=SequenceForecastDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=SequenceForecastDataset.collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=SequenceForecastDataset.collate_fn,
    )
    return train_loader, val_loader, test_loader


def forward_pass(
    model: ForecastModel,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    tokens = batch["tokens"].to(device)
    numeric = batch["numeric"].to(device)
    time_feat = batch["time"].to(device)
    group_id = batch["group_id"].to(device)
    y_class = batch["y_class"].to(device)
    y_rtf = batch["y_rtf"].to(device)
    y_mask = batch["y_mask"].to(device)
    weight = batch["weight"].to(device)

    outputs = model({
        "tokens": tokens,
        "numeric": numeric,
        "time": time_feat,
        "group_id": group_id,
    })
    outputs["y_class"] = y_class
    outputs["y_rtf"] = y_rtf
    outputs["y_mask"] = y_mask
    outputs["weight"] = weight
    return outputs


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    pos_weight: torch.Tensor,
    criterion_reg: nn.Module,
    lambda_reg: float,
    focal_gamma: float,
    focal_alpha: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = outputs["logits"]
    y_class = outputs["y_class"]
    y_rtf = outputs["y_rtf"]
    y_mask = outputs["y_mask"].float()
    weights = outputs["weight"].float()

    # Classification loss with focal loss
    raw_bce = F.binary_cross_entropy_with_logits(logits, y_class, pos_weight=pos_weight, reduction="none")
    if focal_gamma > 0:
        probs = torch.sigmoid(logits)
        pt = torch.where(y_class > 0.5, probs, 1.0 - probs)
        focal_factor = (1.0 - pt).clamp(min=1e-6).pow(focal_gamma)
    else:
        focal_factor = torch.ones_like(raw_bce)

    if 0.0 < focal_alpha < 1.0:
        alpha_pos = torch.full_like(y_class, fill_value=focal_alpha)
        alpha_neg = torch.full_like(y_class, fill_value=1.0 - focal_alpha)
        alpha_factor = torch.where(y_class > 0.5, alpha_pos, alpha_neg)
    else:
        alpha_factor = torch.ones_like(raw_bce)

    raw_loss_class = raw_bce * focal_factor * alpha_factor
    weighted_sum = torch.sum(raw_loss_class * weights)
    normalizer = torch.sum(weights).clamp(min=1.0)
    loss_class = weighted_sum / normalizer
    
    # Regression loss: compute for all masked samples, but weight positive samples more
    if y_mask.sum() > 0:
        mask_bool = y_mask.bool()
        rtf_pred = outputs["rtf"][mask_bool]
        rtf_true = y_rtf[mask_bool]
        
        # Weight regression loss: positive samples get full weight, negative samples get reduced weight
        reg_weights = torch.where(y_class[mask_bool] > 0.5, 
                                  torch.ones_like(rtf_pred), 
                                  torch.full_like(rtf_pred, 0.1))
        
        # Compute weighted regression loss
        reg_loss_raw = F.smooth_l1_loss(rtf_pred, rtf_true, reduction="none")
        reg_loss_weighted = (reg_loss_raw * reg_weights).sum() / reg_weights.sum().clamp(min=1.0)
        loss_reg = reg_loss_weighted
    else:
        loss_reg = torch.tensor(0.0, device=logits.device)
    
    loss = loss_class + lambda_reg * loss_reg
    return loss, {"loss_class": float(loss_class.detach().cpu()), "loss_reg": float(loss_reg.detach().cpu())}


def run_epoch(
    model: ForecastModel,
    data_loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
    criterion_reg: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler | None,
    lambda_reg: float,
    horizon: int,
    focal_gamma: float,
    focal_alpha: float,
    clip_norm: float | None,
    top_fractions: Sequence[float] | None,
    include_curves: bool,
    train: bool = True,
    epoch: int = 0,
    logger: logging.Logger | None = None,
) -> Tuple[float, Dict[str, Any]]:
    model.train(mode=train)
    epoch_loss = 0.0
    total_samples = 0
    logits_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    rtf_pred_list: List[np.ndarray] = []
    rtf_true_list: List[np.ndarray] = []
    rtf_mask_list: List[np.ndarray] = []

    phase = "训练" if train else "验证"
    desc = f"Epoch {epoch} - {phase}"
    
    # 创建进度条
    pbar = tqdm(
        data_loader,
        desc=desc,
        unit="batch",
        leave=False,
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    batch_losses = []
    for batch_idx, batch in enumerate(pbar):
        total_samples += batch["tokens"].size(0)
        if train:
            optimizer.zero_grad(set_to_none=True)

        amp_enabled = scaler is not None and scaler.is_enabled()
        if amp_enabled:
            with autocast():
                outputs = forward_pass(model, batch, device)
                loss, loss_dict = compute_loss(outputs, pos_weight, criterion_reg, lambda_reg, focal_gamma, focal_alpha)
            scaler.scale(loss).backward()
            if train:
                scaler.unscale_(optimizer)
                if clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = forward_pass(model, batch, device)
            loss, loss_dict = compute_loss(outputs, pos_weight, criterion_reg, lambda_reg, focal_gamma, focal_alpha)
            if train:
                loss.backward()
                if clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()

        batch_loss = float(loss.detach().cpu())
        batch_losses.append(batch_loss)
        epoch_loss += batch_loss * batch["tokens"].size(0)

        probs = torch.sigmoid(outputs["logits"]).detach().cpu().numpy()
        logits_list.append(probs)
        labels_list.append(outputs["y_class"].detach().cpu().numpy())
        rtf_pred_list.append(outputs["rtf"].detach().cpu().numpy())
        rtf_true_list.append(outputs["y_rtf"].detach().cpu().numpy())
        rtf_mask_list.append(outputs["y_mask"].detach().cpu().numpy())
        
        # 更新进度条显示
        avg_loss = np.mean(batch_losses[-10:]) if len(batch_losses) > 0 else batch_loss
        loss_class = loss_dict.get("loss_class", 0.0)
        loss_reg = loss_dict.get("loss_reg", 0.0)
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'cls': f'{loss_class:.4f}',
            'reg': f'{loss_reg:.4f}',
            'samples': total_samples
        })
    
    pbar.close()

    metrics = aggregate_metrics(
        logits_list,
        labels_list,
        rtf_pred_list,
        rtf_true_list,
        rtf_mask_list,
        horizon=horizon,
        top_fractions=top_fractions,
        include_curves=include_curves,
    )
    metrics.update({"loss": epoch_loss / max(total_samples, 1)})
    return metrics["loss"], metrics


def aggregate_metrics(
    logits_list: Sequence[np.ndarray],
    labels_list: Sequence[np.ndarray],
    rtf_pred_list: Sequence[np.ndarray],
    rtf_true_list: Sequence[np.ndarray],
    rtf_mask_list: Sequence[np.ndarray],
    horizon: int,
    top_fractions: Sequence[float] | None,
    include_curves: bool,
) -> Dict[str, Any]:
    scores = np.concatenate(logits_list)
    labels = np.concatenate(labels_list)
    rtf_pred = np.concatenate(rtf_pred_list)
    rtf_true = np.concatenate(rtf_true_list)
    rtf_mask = np.concatenate(rtf_mask_list).astype(bool)

    alert_metrics = evaluate_alerts(scores, labels, k_fraction=0.2)
    lead_metrics = compute_lead_time_stats(scores, labels, horizon=horizon)
    reg_metrics = regression_metrics(rtf_pred, rtf_true, mask=rtf_mask)

    multik_metrics = evaluate_multi_thresholds(scores, labels, top_fractions or []) if top_fractions else {}

    metrics = {}
    metrics.update(alert_metrics)
    metrics.update(lead_metrics)
    metrics.update({f"reg_{k}": v for k, v in reg_metrics.items()})
    metrics.update(multik_metrics)

    if include_curves:
        pr = precision_recall_curve(scores, labels)
        roc = roc_curve(scores, labels)
        metrics["pr_curve"] = {
            "precision": pr["precision"].tolist(),
            "recall": pr["recall"].tolist(),
            "auc": pr["auc"],
        }
        metrics["roc_curve"] = {
            "tpr": roc["tpr"].tolist(),
            "fpr": roc["fpr"].tolist(),
            "auc": roc["auc"],
        }
    return metrics


def save_metrics(output_dir: Path, train_hist: List[Dict[str, Any]], val_hist: List[Dict[str, Any]], test_metrics: Dict[str, Any]) -> None:
    payload = {
        "train_history": train_hist,
        "val_history": val_hist,
        "test_metrics": test_metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "forecast_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main(argv: Sequence[str] | None = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Train sequence forecast model for OpenStack logs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-lines-per-file", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--alert-horizon", type=int, default=None)
    parser.add_argument("--normal-downsample", type=float, default=None)
    parsed = parser.parse_args(argv)

    config = ForecastConfig()
    config.resolve_paths()
    if parsed.device:
        config.device = parsed.device
    if parsed.epochs:
        config.epochs = parsed.epochs
    if parsed.batch_size:
        config.batch_size = parsed.batch_size
    if parsed.lr:
        config.learning_rate = parsed.lr
    if parsed.output_dir:
        config.output_dir = Path(parsed.output_dir)
        config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志记录器
    logger = setup_logging(config.output_dir)
    logger.info("=" * 80)
    logger.info("开始训练序列预测模型")
    logger.info("=" * 80)
    if parsed.max_lines_per_file is not None:
        config.max_lines_per_file = parsed.max_lines_per_file
    if parsed.window_size is not None:
        config.window_size = parsed.window_size
    if parsed.window_stride is not None:
        config.window_stride = parsed.window_stride
    if parsed.sequence_length is not None:
        config.sequence_length = parsed.sequence_length
    if parsed.alert_horizon is not None:
        config.alert_horizon = parsed.alert_horizon
    if parsed.normal_downsample is not None:
        config.normal_downsample = parsed.normal_downsample

    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    set_seed(config.seed)
    
    logger.info(f"使用设备: {device}")
    if device.type == "cuda":
        logger.info(f"GPU 名称: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    logger.info(f"随机种子: {config.seed}")
    logger.info("正在加载数据集...")

    train_ds, val_ds, test_ds, vocab = prepare_datasets(config)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_ds,
        val_ds,
        test_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # Compute dataset statistics
    train_labels = np.array([s.y_class for s in train_ds.sequences], dtype=np.float32)
    val_labels = np.array([s.y_class for s in val_ds.sequences], dtype=np.float32)
    test_labels = np.array([s.y_class for s in test_ds.sequences], dtype=np.float32)
    
    logger.info("数据集统计信息:")
    logger.info(f"  训练集: {len(train_ds)} 个样本 (正样本: {train_labels.sum():.0f}/{len(train_labels)}, 比例: {train_labels.mean():.4f})")
    logger.info(f"  验证集: {len(val_ds)} 个样本 (正样本: {val_labels.sum():.0f}/{len(val_labels)}, 比例: {val_labels.mean():.4f})")
    logger.info(f"  测试集: {len(test_ds)} 个样本 (正样本: {test_labels.sum():.0f}/{len(test_labels)}, 比例: {test_labels.mean():.4f})")
    logger.info(f"  词汇表大小: {len(vocab)}")
    
    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty! Check data loading and sequence building.")
    if len(val_ds) == 0:
        raise ValueError("Validation dataset is empty! Check data loading and sequence building.")
    if train_labels.sum() == 0:
        logger.warning("警告: 训练集中没有正样本!")
    if val_labels.sum() == 0:
        logger.warning("警告: 验证集中没有正样本!")

    pos_weight = compute_pos_weight(train_ds)
    pos_weight_tensor = torch.tensor(pos_weight, device=device)
    logger.info(f"正样本权重: {pos_weight:.4f}")
    
    logger.info("正在初始化模型...")
    model = ForecastModel(
        vocab_size=len(vocab),
        num_groups=len(train_ds.group_to_idx),
        embedding_dim=config.embedding_dim,
        numeric_dim=config.numeric_dim,
        time_dim=4,
        id_embedding_dim=config.id_embedding_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        dropout=config.dropout,
    ).to(device)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    criterion_reg = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # Add learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3,
        threshold=0.01, threshold_mode='rel'
    )
    scaler = GradScaler(enabled=config.use_amp and device.type == "cuda")
    
    logger.info("训练配置:")
    logger.info(f"  总轮数: {config.epochs}")
    logger.info(f"  批次大小: {config.batch_size}")
    logger.info(f"  初始学习率: {config.learning_rate}")
    logger.info(f"  权重衰减: {config.weight_decay}")
    logger.info(f"  混合精度训练: {config.use_amp and device.type == 'cuda'}")
    logger.info(f"  梯度裁剪: {config.clip_grad_norm}")
    logger.info(f"  回归损失权重: {config.lambda_reg}")
    logger.info(f"  Focal Loss (gamma): {config.focal_gamma}")
    logger.info(f"  Focal Loss (alpha): {config.focal_alpha}")
    logger.info("=" * 80)

    train_history: List[Dict[str, float]] = []
    val_history: List[Dict[str, float]] = []

    best_score = -math.inf
    best_state: Dict[str, torch.Tensor] | None = None
    training_start = time.time()
    
    logger.info("开始训练循环...")
    logger.info("")

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        _, train_metrics = run_epoch(
            model,
            train_loader,
            device,
            pos_weight_tensor,
            criterion_reg,
            optimizer,
            scaler,
            config.lambda_reg,
            config.alert_horizon,
            config.focal_gamma,
            config.focal_alpha,
            config.clip_grad_norm,
            top_fractions=config.eval_k_fractions,
            include_curves=False,
            train=True,
            epoch=epoch,
            logger=logger,
        )
        train_time = time.time() - epoch_start
        
        val_start = time.time()
        _, val_metrics = run_epoch(
            model,
            val_loader,
            device,
            pos_weight_tensor,
            criterion_reg,
            optimizer,
            scaler=None,
            lambda_reg=config.lambda_reg,
            horizon=config.alert_horizon,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha,
            clip_norm=None,
            top_fractions=config.eval_k_fractions,
            include_curves=False,
            train=False,
            epoch=epoch,
            logger=logger,
        )
        val_time = time.time() - val_start

        train_metrics["epoch"] = epoch
        val_metrics["epoch"] = epoch
        train_history.append(train_metrics)
        val_history.append(val_metrics)

        score = val_metrics.get("recall@top", 0.0)
        # Also consider AUPRC if available
        if "pr_curve" in val_metrics:
            score = max(score, val_metrics["pr_curve"].get("auc", 0.0))
        
        # Update learning rate based on validation score
        old_lr = current_lr
        scheduler.step(score)
        new_lr = optimizer.param_groups[0]['lr']
        lr_changed = old_lr != new_lr
        
        is_best = score > best_score
        if is_best:
            best_score = score
            best_state = model.state_dict()

        elapsed = time.time() - training_start
        remaining_epochs = max(config.epochs - epoch, 0)
        avg_epoch_time = elapsed / epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_str = str(timedelta(seconds=int(eta_seconds)))

        # 详细的日志输出
        logger.info(f"Epoch {epoch}/{config.epochs} 完成")
        logger.info(f"  训练阶段:")
        logger.info(f"    损失: {train_metrics['loss']:.6f} (分类: {train_metrics.get('loss_class', 0):.6f}, 回归: {train_metrics.get('loss_reg', 0):.6f})")
        logger.info(f"    时间: {train_time:.2f}秒")
        logger.info(f"  验证阶段:")
        logger.info(f"    损失: {val_metrics['loss']:.6f} (分类: {val_metrics.get('loss_class', 0):.6f}, 回归: {val_metrics.get('loss_reg', 0):.6f})")
        logger.info(f"    Recall@Top: {val_metrics.get('recall@top', 0):.6f}")
        logger.info(f"    Precision@Top: {val_metrics.get('precision@top', 0):.6f}")
        logger.info(f"    回归 MAE: {val_metrics.get('reg_mae', 0):.6f}")
        logger.info(f"    回归 RMSE: {val_metrics.get('reg_rmse', 0):.6f}")
        logger.info(f"    时间: {val_time:.2f}秒")
        logger.info(f"  学习率: {old_lr:.2e}" + (f" -> {new_lr:.2e} (已降低)" if lr_changed else ""))
        logger.info(f"  当前最佳分数: {best_score:.6f}" + (" [新最佳!]" if is_best else ""))
        logger.info(f"  已用时间: {str(timedelta(seconds=int(elapsed)))}")
        logger.info(f"  预计剩余: {eta_str}")
        logger.info("-" * 80)

    total_training_time = time.time() - training_start
    logger.info("")
    logger.info("=" * 80)
    logger.info("训练完成!")
    logger.info(f"总训练时间: {str(timedelta(seconds=int(total_training_time)))}")
    logger.info(f"平均每轮时间: {total_training_time / config.epochs:.2f}秒")
    logger.info("")
    logger.info("正在加载最佳模型进行测试...")
    
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"已加载最佳模型 (验证分数: {best_score:.6f})")

    logger.info("正在评估测试集...")
    _, test_metrics = run_epoch(
        model,
        test_loader,
        device,
        pos_weight_tensor,
        criterion_reg,
        optimizer,
        scaler=None,
        lambda_reg=config.lambda_reg,
        horizon=config.alert_horizon,
        focal_gamma=config.focal_gamma,
        focal_alpha=config.focal_alpha,
        clip_norm=None,
        top_fractions=config.eval_k_fractions,
        include_curves=True,
        train=False,
        epoch=config.epochs + 1,
        logger=logger,
    )

    logger.info("")
    logger.info("测试集评估结果:")
    logger.info(f"  损失: {test_metrics['loss']:.6f}")
    logger.info(f"  Recall@Top: {test_metrics.get('recall@top', 0):.6f}")
    logger.info(f"  Precision@Top: {test_metrics.get('precision@top', 0):.6f}")
    logger.info(f"  回归 MAE: {test_metrics.get('reg_mae', 0):.6f}")
    logger.info(f"  回归 RMSE: {test_metrics.get('reg_rmse', 0):.6f}")
    if "pr_curve" in test_metrics:
        logger.info(f"  PR-AUC: {test_metrics['pr_curve'].get('auc', 0):.6f}")
    if "roc_curve" in test_metrics:
        logger.info(f"  ROC-AUC: {test_metrics['roc_curve'].get('auc', 0):.6f}")
    
    # 诊断信息
    logger.info("")
    logger.info("诊断信息:")
    if test_metrics['loss'] == 0.0:
        logger.warning("  警告: 损失为0，这很不正常！可能的原因:")
        logger.warning("    - 损失计算有bug")
        logger.warning("    - 测试集为空或样本数计算错误")
        logger.warning("    - 模型预测过于完美（不太可能）")
    if test_metrics.get('precision@top', 0) == 1.0 and test_metrics.get('recall@top', 0) < 0.5:
        logger.warning("  警告: Precision=1.0但Recall较低，可能的原因:")
        logger.warning("    - 测试集中正样本数量很少")
        logger.warning("    - 数据分布极不平衡")
        logger.warning("    - 模型预测过于保守")
    if test_metrics.get('pr_curve', {}).get('auc', 0) == 1.0 and test_metrics.get('roc_curve', {}).get('auc', 0) == 1.0:
        logger.warning("  警告: AUC都是1.0，这通常意味着:")
        logger.warning("    - 测试集中只有一个类别（更可能）")
        logger.warning("    - 数据或指标计算有问题")
    
    # 检查测试集标签分布
    test_labels_array = np.array([s.y_class for s in test_ds.sequences], dtype=np.float32)
    test_pos_ratio = test_labels_array.mean()
    logger.info(f"  测试集正样本比例: {test_pos_ratio:.4f} ({test_labels_array.sum():.0f}/{len(test_labels_array)})")
    if test_pos_ratio == 0.0:
        logger.warning("  警告: 测试集中没有正样本！")
    elif test_pos_ratio < 0.01:
        logger.warning("  警告: 测试集中正样本比例极低！")
    
    logger.info("")
    logger.info("正在保存模型和指标...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab.token_to_idx,
        "config": config.__dict__,
    }, config.output_dir / "forecast_model.pt")
    logger.info(f"模型已保存到: {config.output_dir / 'forecast_model.pt'}")

    save_metrics(config.output_dir, train_history, val_history, test_metrics)
    logger.info(f"指标已保存到: {config.output_dir / 'forecast_metrics.json'}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("所有任务完成!")
    logger.info("=" * 80)
    
    return test_metrics


if __name__ == "__main__":
     main()
