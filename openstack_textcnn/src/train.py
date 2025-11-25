"""训练脚本：基于轻量级 TextCNN 的 OpenStack 日志异常检测实现。

本脚本严格对应《具体方案》中提出的 TextCNN 思路：
- 使用模板化预处理（见 data.py）获取轻量级输入特征；
- 采用单层多卷积核 + 全局最大池化的 TextCNN（见 model.py）；
- 输出 DevOps 流水线可消费的 metrics.json、混淆矩阵与显著性解释结果。
"""

# 导入命令行参数解析模块
import argparse
# 导入JSON处理模块，用于保存和加载配置、指标
import json
# 导入随机数模块，用于设置随机种子
import random
# 导入路径处理模块，用于文件路径操作
from pathlib import Path
# 导入类型注解，用于函数类型提示
from typing import Dict, Optional, Tuple

# 导入NumPy数值计算库
import numpy as np
# 导入PyTorch深度学习框架
import torch
# 导入sklearn评估指标模块
from sklearn.metrics import (
    accuracy_score,  # 准确率计算
    confusion_matrix,  # 混淆矩阵计算
    precision_recall_fscore_support,  # 精确率、召回率、F1分数计算
    roc_auc_score,  # ROC-AUC分数计算
)
# 导入PyTorch神经网络模块和优化器
from torch import nn, optim
# 导入数据加载器和加权随机采样器
from torch.utils.data import DataLoader, WeightedRandomSampler
# 导入进度条显示库
from tqdm import tqdm

# 导入自定义模块：数据集划分、日志数据集、数值特征维度、词汇表、数据加载和划分函数
from .data import DatasetSplits, LogDataset, NUMERIC_FEATURE_DIM, Vocabulary, load_dataset, split_dataset
# 导入TextCNN模型
from .model import TextCNN
# 导入报告生成函数：混淆矩阵保存和显著性报告保存
from .reporting import save_confusion_matrix, save_saliency_report


# ============================================================================
# 随机种子设置：确保实验可复现
# ============================================================================
def set_seed(seed: int) -> None:
    """设置所有随机数生成器的种子，确保实验可复现"""
    random.seed(seed)  # Python内置random模块的种子
    np.random.seed(seed)  # NumPy随机数生成器的种子
    torch.manual_seed(seed)  # PyTorch CPU随机数生成器的种子
    torch.cuda.manual_seed_all(seed)  # PyTorch所有GPU随机数生成器的种子


# ============================================================================
# 数据加载器构建：创建训练、验证、测试数据加载器
# ============================================================================
def build_dataloaders(
    splits: DatasetSplits,  # 数据集划分（训练集、验证集、测试集）
    vocab: Vocabulary,  # 词汇表对象
    batch_size: int,  # 批次大小
    max_length: int,  # 最大序列长度
    balance_sampler: bool,  # 是否使用类别平衡采样器
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练、验证、测试数据加载器"""
    # 创建训练数据集：将DataFrame转换为PyTorch数据集
    train_ds = LogDataset(splits.train, vocab, max_length=max_length)
    # 创建验证数据集
    val_ds = LogDataset(splits.val, vocab, max_length=max_length)
    # 创建测试数据集
    test_ds = LogDataset(splits.test, vocab, max_length=max_length)

    # ========================================================================
    # 类别平衡采样：处理类别不平衡问题
    # ========================================================================
    if balance_sampler:  # 如果启用类别平衡采样
        # 统计每个类别的样本数量
        class_counts = splits.train["label"].value_counts().to_dict()
        # 计算每个样本的采样权重：类别样本数越少，权重越大
        # 权重 = 1 / 类别样本数，使得少数类样本被更频繁地采样
        class_weights = splits.train["label"].map(lambda lbl: 1.0 / class_counts[int(lbl)])
        # 将权重转换为PyTorch张量
        weights_tensor = torch.as_tensor(class_weights.to_numpy(dtype=np.float64), dtype=torch.double)
        # 创建加权随机采样器：根据权重随机采样，允许重复采样（replacement=True）
        sampler = WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights_tensor), replacement=True)
        # 使用采样器创建训练数据加载器（不使用shuffle，因为采样器已处理随机性）
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, collate_fn=LogDataset.collate)
    else:  # 如果不使用类别平衡采样
        # 使用标准随机打乱创建训练数据加载器
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=LogDataset.collate)
    # 验证集和测试集不需要打乱，保持数据顺序
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=LogDataset.collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=LogDataset.collate)
    return train_loader, val_loader, test_loader


# ============================================================================
# 正样本权重计算：用于处理类别不平衡的损失函数
# ============================================================================
def compute_pos_weight(df) -> float:
    """计算正样本权重，用于BCEWithLogitsLoss的pos_weight参数"""
    # 统计正样本（异常样本，label=1）的数量，至少为1避免除零
    positives = max(df["label"].sum(), 1)
    # 统计负样本（正常样本，label=0）的数量，至少为1避免除零
    negatives = max(len(df) - positives, 1)
    # 返回负样本数/正样本数，用于在损失函数中平衡两类样本的贡献
    return negatives / positives


# ============================================================================
# 训练一个epoch：完成一轮完整的训练过程
# ============================================================================
def train_epoch(model, data_loader, criterion, optimizer, device) -> float:
    """训练模型一个epoch，返回平均损失"""
    # 将模型设置为训练模式：启用Dropout、BatchNorm等训练时的行为
    model.train()
    epoch_loss = 0.0  # 初始化epoch总损失
    # 遍历数据加载器中的每个批次，显示进度条
    for inputs, numeric_features, labels in tqdm(data_loader, desc="train", leave=False):
        # 将输入数据移动到指定设备（CPU或GPU）
        inputs = inputs.to(device)
        numeric_features = numeric_features.to(device)
        labels = labels.to(device)
        # 清零梯度：每次迭代前清除上一次的梯度，避免梯度累积
        optimizer.zero_grad()
        # 前向传播：模型预测，得到logits（未归一化的分数）
        logits = model(inputs, numeric_features)
        # 计算损失：使用损失函数计算预测值与真实值的差异
        loss = criterion(logits, labels)
        # 反向传播：计算梯度，从输出层向输入层传播误差
        loss.backward()
        # 更新参数：根据梯度更新模型参数
        optimizer.step()
        # 累加损失：将批次损失乘以批次大小，得到总损失
        epoch_loss += loss.item() * len(inputs)
    # 返回平均损失：总损失除以数据集大小
    return epoch_loss / len(data_loader.dataset)


# ============================================================================
# 模型评估：在验证集或测试集上评估模型性能
# ============================================================================
@torch.no_grad()  # 装饰器：禁用梯度计算，节省内存和加速推理
def evaluate(model, data_loader, criterion, device):
    """评估模型，返回损失、概率和真实标签"""
    # 将模型设置为评估模式：禁用Dropout、使用BatchNorm的统计量
    model.eval()
    all_logits = []  # 存储所有批次的logits
    all_labels = []  # 存储所有批次的真实标签
    loss_total = 0.0  # 初始化总损失
    # 遍历数据加载器中的每个批次
    for inputs, numeric_features, labels in tqdm(data_loader, desc="eval", leave=False):
        # 将数据移动到指定设备
        inputs = inputs.to(device)
        numeric_features = numeric_features.to(device)
        labels = labels.to(device)
        # 前向传播：获取模型预测的logits
        logits = model(inputs, numeric_features)
        # 计算损失
        loss = criterion(logits, labels)
        # 累加损失
        loss_total += loss.item() * len(inputs)
        # 将logits和标签移到CPU并断开梯度连接，添加到列表
        all_logits.append(logits.detach().cpu())  # detach()断开梯度，cpu()移到CPU
        all_labels.append(labels.detach().cpu())

    # 拼接所有批次的logits和标签，转换为numpy数组
    logits = torch.cat(all_logits).numpy()  # 在批次维度拼接
    labels = torch.cat(all_labels).numpy()  # 在批次维度拼接
    # 计算平均损失
    loss = loss_total / len(labels)
    # 将logits通过sigmoid函数转换为概率（0-1之间）
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    return loss, probs, labels


# ============================================================================
# 指标计算：根据概率和阈值计算各种评估指标
# ============================================================================
def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Tuple[Dict[str, float], np.ndarray]:
    """计算分类指标，返回指标字典和预测结果"""
    # 根据阈值将概率转换为二分类预测（0或1）
    # probs >= threshold 返回布尔数组，astype(int)转换为0/1
    preds = (probs >= threshold).astype(int)
    # 计算准确率：正确预测的样本数 / 总样本数
    acc = accuracy_score(labels, preds)
    # 计算精确率、召回率、F1分数（二分类，zero_division=0表示除零时返回0）
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    # 尝试计算ROC-AUC分数（需要至少两个类别都有样本）
    try:
        auc = roc_auc_score(labels, probs)  # 使用概率而非预测值计算AUC
    except ValueError:  # 如果只有一个类别（如验证集全是正样本），捕获异常
        auc = float("nan")  # 返回NaN

    # 计算混淆矩阵：[[TN, FP], [FN, TP]]，转换为列表格式
    cm = confusion_matrix(labels, preds).astype(int).tolist()
    # 处理AUC值：如果是NaN则设为None，否则转换为float
    auc_value = None if np.isnan(auc) else float(auc)
    # 构建指标字典
    metrics = {
        "loss": float("nan"),  # 损失值（在此函数中不计算，由调用者设置）
        "accuracy": float(acc),  # 准确率
        "precision": float(precision),  # 精确率：预测为正样本中真正为正的比例
        "recall": float(recall),  # 召回率：真正为正样本中被正确预测的比例
        "f1": float(f1),  # F1分数：精确率和召回率的调和平均
        "auc": auc_value,  # ROC-AUC：ROC曲线下面积，衡量分类器整体性能
        "confusion_matrix": cm,  # 混淆矩阵：详细展示分类结果
        "threshold": threshold,  # 使用的决策阈值
    }
    return metrics, preds


# ============================================================================
# 最优阈值搜索：在验证集上寻找使F1分数最大的决策阈值
# ============================================================================
def find_best_threshold(probs: np.ndarray, labels: np.ndarray, base_threshold: float) -> Tuple[float, Dict[str, float]]:
    """在候选阈值中搜索使F1分数最大的阈值"""
    # 生成均匀分布的候选阈值：从0.1到0.9，共17个点
    grid = np.linspace(0.1, 0.9, num=17)
    # 生成基于概率分位数的候选阈值：从5%到95%分位数，共19个点
    # 这些阈值更可能接近实际最优值
    quantiles = np.quantile(probs, np.linspace(0.05, 0.95, num=19))
    # 合并所有候选阈值：基础阈值、均匀网格、分位数阈值，去重并排序
    candidate_thresholds = np.unique(np.concatenate(([base_threshold], grid, quantiles)))

    # 初始化最佳阈值和最佳指标
    best_threshold = base_threshold
    best_metrics, _ = compute_metrics(probs, labels, best_threshold)

    # 遍历所有候选阈值，寻找F1分数最大的阈值
    for threshold in candidate_thresholds:
        metrics, _ = compute_metrics(probs, labels, threshold)
        # 如果当前阈值的F1分数更高，更新最佳阈值和指标
        if metrics["f1"] > best_metrics["f1"]:
            best_metrics = metrics
            best_threshold = threshold

    return best_threshold, best_metrics


# ============================================================================
# 主训练流程：完整的模型训练、验证、测试和结果保存
# ============================================================================
def run_training(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    """执行完整的训练流程，返回训练指标"""
    # ========================================================================
    # 路径设置和随机种子初始化
    # ========================================================================
    # 获取数据目录路径
    data_dir = Path(args.data_dir)
    # 获取输出目录路径
    output_dir = Path(args.output_dir)
    # 创建输出目录（如果不存在），parents=True表示创建父目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置随机种子，确保实验可复现
    set_seed(args.seed)

    # ========================================================================
    # 数据加载和预处理
    # ========================================================================
    # 加载数据集：从日志文件中读取数据并进行预处理
    df = load_dataset(
        data_dir,  # 数据目录
        max_lines_per_file=args.max_lines,  # 每个文件最大读取行数（None表示全部）
        normal_downsample=args.normal_downsample,  # 正常样本下采样比例（处理类别不平衡）
        deduplicate=args.deduplicate,  # 是否去重
        window_size=args.window_size,  # 滑动窗口大小（用于聚合上下文）
        window_stride=args.window_stride,  # 滑动窗口步长
    )
    # 划分数据集：将数据分为训练集、验证集、测试集
    splits = split_dataset(df, train_ratio=args.train_ratio, val_ratio=args.val_ratio, random_state=args.seed)

    # 辅助函数：统计类别分布
    def _count(series):
        """统计Series中每个值的出现次数，返回字典"""
        return {int(k): int(v) for k, v in series.value_counts().to_dict().items()}

    # 构建数据集摘要信息：用于记录和报告
    dataset_summary = {
        "total_records": int(len(df)),  # 总记录数
        "train_size": int(len(splits.train)),  # 训练集大小
        "val_size": int(len(splits.val)),  # 验证集大小
        "test_size": int(len(splits.test)),  # 测试集大小
        "class_distribution": {  # 各类别在不同数据集中的分布
            "train": _count(splits.train["label"]),  # 训练集类别分布
            "val": _count(splits.val["label"]),  # 验证集类别分布
            "test": _count(splits.test["label"]),  # 测试集类别分布
        },
    }

    # ========================================================================
    # 词汇表构建
    # ========================================================================
    # 创建词汇表对象：设置最小词频和最大词汇表大小
    vocab = Vocabulary(min_freq=args.min_freq, max_size=args.max_vocab)
    # 从训练集文本中构建词汇表（只使用训练集，避免数据泄露）
    vocab.build(splits.train["text"].tolist())

    # ========================================================================
    # 数据加载器构建
    # ========================================================================
    # 创建训练、验证、测试数据加载器
    train_loader, val_loader, test_loader = build_dataloaders(
        splits,  # 数据集划分
        vocab,  # 词汇表
        batch_size=args.batch_size,  # 批次大小
        max_length=args.max_length,  # 最大序列长度
        balance_sampler=args.balance_sampler,  # 是否使用类别平衡采样
    )

    # ========================================================================
    # 模型初始化
    # ========================================================================
    # 选择计算设备：如果有GPU且未强制使用CPU，则使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    # 创建TextCNN模型实例
    model = TextCNN(
        vocab_size=len(vocab),  # 词汇表大小
        embedding_dim=args.embedding_dim,  # 词嵌入维度
        num_filters=args.num_filters,  # 卷积滤波器数量
        kernel_sizes=tuple(args.kernel_sizes),  # 卷积核尺寸元组
        dropout=args.dropout,  # Dropout比率
        hidden_dim=args.hidden_dim,  # 全连接层隐藏维度
        numeric_feature_dim=NUMERIC_FEATURE_DIM,  # 数值特征维度
        numeric_hidden_dim=args.numeric_hidden_dim,  # 数值特征编码器隐藏维度
    ).to(device)  # 将模型移动到指定设备

    # ========================================================================
    # 损失函数和优化器设置
    # ========================================================================
    # 计算正样本权重：如果未手动指定，则根据训练集类别分布自动计算
    pos_weight_value = args.pos_weight if args.pos_weight > 0 else compute_pos_weight(splits.train)
    # 创建带logits的二元交叉熵损失函数，pos_weight用于平衡类别不平衡
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    # 创建Adam优化器：自适应学习率，weight_decay用于L2正则化
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ========================================================================
    # 训练循环初始化
    # ========================================================================
    best_val_f1 = 0.0  # 最佳验证集F1分数
    best_threshold = float(args.threshold)  # 最佳决策阈值
    best_val_metrics: Optional[Dict[str, float]] = None  # 最佳验证集指标
    best_model_path = output_dir / "best_model.pt"  # 最佳模型保存路径
    history = {"train": [], "val": []}  # 训练历史记录

    # ========================================================================
    # 训练循环：迭代训练多个epoch
    # ========================================================================
    for epoch in range(1, args.epochs + 1):
        # 训练一个epoch：在训练集上更新模型参数
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        # 在验证集上评估模型性能
        val_loss, val_probs, val_labels = evaluate(model, val_loader, criterion, device)

        # ====================================================================
        # 阈值选择：根据配置选择固定阈值或优化阈值
        # ====================================================================
        if args.threshold_mode == "optimize":  # 如果使用优化模式
            # 在验证集上搜索使F1分数最大的阈值
            threshold_candidate, metrics_candidate = find_best_threshold(val_probs, val_labels, best_threshold)
        else:  # 如果使用固定模式
            # 使用命令行指定的固定阈值
            threshold_candidate = float(args.threshold)
            # 使用固定阈值计算指标
            metrics_candidate, _ = compute_metrics(val_probs, val_labels, threshold_candidate)

        # 更新指标字典：添加损失和阈值信息
        metrics_candidate = dict(metrics_candidate)
        metrics_candidate["loss"] = float(val_loss)  # 添加验证损失
        metrics_candidate["threshold"] = float(threshold_candidate)  # 添加使用的阈值

        # 记录训练历史：保存每个epoch的训练和验证指标
        history["train"].append({"epoch": epoch, "loss": float(train_loss)})
        history["val"].append({"epoch": epoch, **metrics_candidate})  # **展开字典

        # 打印当前epoch的训练信息
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_f1={metrics_candidate['f1']:.4f} "
            f"threshold={threshold_candidate:.3f}"
        )

        # ====================================================================
        # 模型保存：如果当前epoch的F1分数更高，保存最佳模型
        # ====================================================================
        if metrics_candidate["f1"] > best_val_f1:  # 如果当前F1分数更高
            best_val_f1 = metrics_candidate["f1"]  # 更新最佳F1分数
            best_threshold = float(threshold_candidate)  # 更新最佳阈值
            best_val_metrics = dict(metrics_candidate)  # 更新最佳指标
            # 保存模型检查点：包含模型参数、词汇表、配置和阈值
            torch.save({
                "model_state_dict": model.state_dict(),  # 模型参数
                "vocab": vocab.token_to_idx,  # 词汇表映射
                "config": vars(args),  # 训练配置（命令行参数）
                "threshold": best_threshold,  # 最佳阈值
            }, best_model_path)

    # ========================================================================
    # 确保模型已保存：如果训练过程中没有保存模型（所有epoch都较差），保存最后一个模型
    # ========================================================================
    if not best_model_path.exists():
        # 保存最后一个epoch的模型
        torch.save({
            "model_state_dict": model.state_dict(),
            "vocab": vocab.token_to_idx,
            "config": vars(args),
            "threshold": best_threshold,
        }, best_model_path)
        # 使用最后一个epoch的验证指标作为最佳指标
        best_val_metrics = history["val"][-1]

    # ========================================================================
    # 加载最佳模型并在测试集上评估
    # ========================================================================
    # 加载保存的最佳模型检查点
    checkpoint = torch.load(best_model_path, map_location=device)
    # 将模型参数加载到当前模型
    model.load_state_dict(checkpoint["model_state_dict"])
    # 从检查点获取最佳阈值（如果存在），否则使用当前值
    best_threshold = float(checkpoint.get("threshold", best_threshold))

    # 在测试集上评估最佳模型
    test_loss, test_probs, test_labels = evaluate(model, test_loader, criterion, device)
    # 使用最佳阈值计算测试集指标
    test_metrics, test_preds = compute_metrics(test_probs, test_labels, best_threshold)
    # 添加测试损失和阈值到指标字典
    test_metrics["loss"] = float(test_loss)
    test_metrics["threshold"] = float(best_threshold)

    # 如果最佳验证指标未设置，使用最后一个epoch的验证指标
    if best_val_metrics is None and history["val"]:
        best_val_metrics = dict(history["val"][-1])

    # ========================================================================
    # 生成可视化报告：混淆矩阵和显著性解释
    # ========================================================================
    # 保存混淆矩阵图像：展示分类结果的详细情况
    save_confusion_matrix(
        test_metrics["confusion_matrix"],  # 混淆矩阵数据
        labels=["normal", "anomaly"],  # 类别标签
        output_path=output_dir / "confusion_matrix.png",  # 输出路径
    )

    # 获取测试集文本，用于显著性分析
    test_texts = splits.test["text"].tolist()
    # 找到概率最高的top_k个样本（最可能为异常的样本）
    # np.argsort(-test_probs)按概率降序排序，取前top_k个索引
    top_indices = np.argsort(-test_probs)[: args.explain_top_k]
    # 提取对应的文本样本
    top_samples = [test_texts[i] for i in top_indices]
    # 生成显著性报告：分析哪些token对预测结果贡献最大
    save_saliency_report(
        model=model,  # 训练好的模型
        vocab=vocab,  # 词汇表
        samples=top_samples,  # 要分析的样本
        device=device,  # 计算设备
        max_length=args.max_length,  # 最大序列长度
        output_path=output_dir / "saliency_report.json",  # 输出路径
        top_k=args.explain_top_k,  # 保存的样本数量
    )

    # ========================================================================
    # 汇总所有指标并保存
    # ========================================================================
    # 构建完整的指标字典
    metrics = {
        "dataset": dataset_summary,  # 数据集摘要信息
        "train_history": history,  # 训练历史（每个epoch的指标）
        "val_best_f1": best_val_f1,  # 最佳验证集F1分数
        "val_best_threshold": best_threshold,  # 最佳阈值
        "val_best_metrics": best_val_metrics,  # 最佳验证集指标
        "decision_threshold": best_threshold,  # 决策阈值（与val_best_threshold相同）
        "test": test_metrics,  # 测试集指标
    }

    # 保存测试集预测结果：概率、预测、标签和阈值
    np.savez(
        output_dir / "test_predictions.npz",  # NumPy压缩格式，可存储多个数组
        probs=test_probs,  # 预测概率
        preds=test_preds,  # 预测结果（0/1）
        labels=test_labels,  # 真实标签
        threshold=best_threshold,  # 使用的阈值
    )

    # 保存指标到JSON文件：用于后续分析和报告
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)  # indent=2使JSON格式更易读

    return metrics


# ============================================================================
# 命令行参数解析：定义训练脚本的所有可配置参数
# ============================================================================
def parse_args() -> argparse.Namespace:
    """解析命令行参数，返回参数命名空间"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Train TextCNN for OpenStack log anomaly detection")
    
    # ========================================================================
    # 数据相关参数
    # ========================================================================
    # 数据目录路径：默认使用项目根目录下的OpenStackData文件夹
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "OpenStackData"))
    # 输出目录：保存模型、指标、报告等
    parser.add_argument("--output-dir", type=str, default="artifacts")
    # 每个文件最大读取行数：None表示读取全部
    parser.add_argument("--max-lines", type=int, default=None)
    # 正常样本下采样比例：用于处理类别不平衡（0.5表示只使用50%的正常样本）
    parser.add_argument("--normal-downsample", type=float, default=0.5)
    # 是否去重：去除重复的文本样本
    parser.add_argument("--deduplicate", action="store_true")
    # 滑动窗口大小：用于聚合上下文信息
    parser.add_argument("--window-size", type=int, default=8)
    # 滑动窗口步长：窗口移动的步长
    parser.add_argument("--window-stride", type=int, default=4)
    
    # ========================================================================
    # 数据集划分参数
    # ========================================================================
    # 训练集比例
    parser.add_argument("--train-ratio", type=float, default=0.7)
    # 验证集比例
    parser.add_argument("--val-ratio", type=float, default=0.15)
    
    # ========================================================================
    # 模型架构参数
    # ========================================================================
    # 词嵌入维度：每个token的向量维度
    parser.add_argument("--embedding-dim", type=int, default=128)
    # 卷积滤波器数量：每个卷积核产生的特征图数量
    parser.add_argument("--num-filters", type=int, default=128)
    # 卷积核尺寸：可以指定多个尺寸，捕获不同长度的模式
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[3, 4, 5])
    # Dropout比率：防止过拟合
    parser.add_argument("--dropout", type=float, default=0.5)
    # 全连接层隐藏维度
    parser.add_argument("--hidden-dim", type=int, default=128)
    # 数值特征编码器隐藏维度
    parser.add_argument("--numeric-hidden-dim", type=int, default=32)
    
    # ========================================================================
    # 训练超参数
    # ========================================================================
    # 训练轮数（epochs）
    parser.add_argument("--epochs", type=int, default=5)
    # 批次大小：每次训练使用的样本数
    parser.add_argument("--batch-size", type=int, default=64)
    # 学习率：控制参数更新的步长
    parser.add_argument("--lr", type=float, default=1e-3)
    # 权重衰减：L2正则化系数，防止过拟合
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    # 正样本权重：<=0表示自动计算，>0表示手动指定
    parser.add_argument("--pos-weight", type=float, default=0.0, help="Manual positive class weight (<=0 to auto-compute).")
    # 是否使用类别平衡采样器：处理类别不平衡问题
    parser.add_argument("--balance-sampler", action="store_true", help="Use class-balanced sampling for training batches.")
    
    # ========================================================================
    # 词汇表参数
    # ========================================================================
    # 最小词频：低于此频率的词将被忽略
    parser.add_argument("--min-freq", type=int, default=2)
    # 最大词汇表大小：限制词汇表大小，避免过大
    parser.add_argument("--max-vocab", type=int, default=20000)
    # 最大序列长度：超过此长度的序列将被截断
    parser.add_argument("--max-length", type=int, default=200)
    
    # ========================================================================
    # 阈值相关参数
    # ========================================================================
    # 阈值选择模式：fixed（固定）或optimize（优化）
    parser.add_argument(
        "--threshold-mode",
        choices=["fixed", "optimize"],
        default="optimize",
        help="Threshold selection strategy (fixed uses --threshold).",
    )
    # 初始或固定决策阈值：用于二分类的阈值
    parser.add_argument("--threshold", type=float, default=0.5, help="Initial or fixed decision threshold.")
    
    # ========================================================================
    # 其他参数
    # ========================================================================
    # 随机种子：确保实验可复现
    parser.add_argument("--seed", type=int, default=42)
    # 强制使用CPU：即使有GPU也不使用
    parser.add_argument("--cpu", action="store_true")
    # 显著性解释的top-k样本数：分析概率最高的k个样本
    parser.add_argument("--explain-top-k", type=int, default=3)
    
    # 解析命令行参数并返回
    return parser.parse_args()


# ============================================================================
# 主程序入口：执行训练流程
# ============================================================================
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    # 执行训练流程
    metrics = run_training(args)
    # 打印测试集指标（JSON格式，便于查看）
    print(json.dumps(metrics["test"], indent=2))
