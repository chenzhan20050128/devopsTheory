# 导入JSON处理模块，用于保存显著性报告
import json
# 导入路径处理模块
from pathlib import Path
# 导入类型注解
from typing import Dict, Iterable, List, Sequence

# 导入matplotlib绘图库，用于绘制混淆矩阵
import matplotlib.pyplot as plt
# 导入NumPy数值计算库
import numpy as np
# 导入PyTorch深度学习框架
import torch

# 导入自定义模块
from .data import Vocabulary  # 词汇表类
from .model import TextCNN  # TextCNN模型类


# ============================================================================
# 混淆矩阵可视化：生成并保存混淆矩阵图像
# ============================================================================
def save_confusion_matrix(cm: Sequence[Sequence[int]], labels: Sequence[str], output_path: Path) -> None:
    """保存混淆矩阵为图像文件"""
    # 将混淆矩阵转换为NumPy数组，便于处理
    matrix = np.array(cm)
    # 创建图形和坐标轴：figsize设置图像大小（宽，高）
    fig, ax = plt.subplots(figsize=(5, 4))
    # 使用热力图显示混淆矩阵：cmap="Blues"使用蓝色配色方案
    im = ax.imshow(matrix, cmap="Blues")

    # 设置x轴和y轴的刻度位置：在矩阵的每个单元格中心
    ax.set_xticks(np.arange(len(labels)))  # x轴刻度位置
    ax.set_yticks(np.arange(len(labels)))  # y轴刻度位置
    # 设置x轴和y轴的刻度标签：类别名称
    ax.set_xticklabels(labels)  # x轴标签（预测类别）
    ax.set_yticklabels(labels)  # y轴标签（真实类别）
    # 设置坐标轴标签
    ax.set_xlabel("Predicted")  # x轴标签：预测值
    ax.set_ylabel("Actual")  # y轴标签：真实值
    # 设置图像标题
    ax.set_title("Confusion Matrix")

    # 在矩阵的每个单元格中显示数值
    for i in range(matrix.shape[0]):  # 遍历行
        for j in range(matrix.shape[1]):  # 遍历列
            # 在位置(j, i)显示矩阵值，居中对齐，黑色文字
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")

    # 调整布局：使图像紧凑，避免标签被裁剪
    fig.tight_layout()
    # 添加颜色条：显示数值与颜色的对应关系
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # 创建输出目录（如果不存在）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 保存图像到文件
    fig.savefig(output_path)
    # 关闭图形，释放内存
    plt.close(fig)


# ============================================================================
# Token准备：将文本转换为模型输入格式
# ============================================================================
def _prepare_tokens(vocab: Vocabulary, text: str, max_length: int) -> tuple[List[int], int]:
    """将文本编码为token ID序列，并进行填充或截断"""
    # 使用词汇表将文本编码为token ID列表
    token_ids = vocab.encode(text)
    # 计算实际使用的长度（不超过max_length）
    length = min(len(token_ids), max_length)
    # 如果序列长度超过最大长度，进行截断
    if len(token_ids) >= max_length:
        tokens = token_ids[:max_length]  # 取前max_length个token
    else:  # 如果序列长度不足，进行填充
        # 使用0（padding token）填充到max_length
        tokens = token_ids + [0] * (max_length - len(token_ids))
    return tokens, length  # 返回token序列和实际长度


# ============================================================================
# 显著性计算：使用梯度方法分析每个token对预测结果的贡献
# ============================================================================
def compute_saliency(
    model: TextCNN,  # 训练好的模型
    vocab: Vocabulary,  # 词汇表
    text: str,  # 要分析的文本
    device: torch.device,  # 计算设备
    max_length: int,  # 最大序列长度
) -> Dict[str, float]:
    """计算每个token对预测结果的显著性（重要性）"""
    # 将模型设置为评估模式：禁用Dropout等训练时的行为
    model.eval()
    # 准备token序列：编码文本并进行填充/截断
    token_ids, length = _prepare_tokens(vocab, text, max_length)

    # ========================================================================
    # 前向传播：获取模型预测
    # ========================================================================
    # 将token ID转换为张量：添加batch维度，形状为(1, seq_len)
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    # 清零梯度：清除之前的梯度信息
    model.zero_grad()
    # 通过嵌入层获取嵌入表示：形状为(1, seq_len, embedding_dim)
    embedded = model.embedding(input_tensor)
    # 保留嵌入层的梯度：retain_grad()允许对中间变量计算梯度
    embedded.retain_grad()
    # 使用forward_with_embeddings进行前向传播：跳过嵌入层，直接使用嵌入表示
    # 转置维度以适应卷积层输入：(1, embedding_dim, seq_len)
    logits = model.forward_with_embeddings(embedded.transpose(1, 2))
    # 将logits转换为概率：使用sigmoid函数
    prob = torch.sigmoid(logits)

    # ========================================================================
    # 反向传播：计算梯度
    # ========================================================================
    # 反向传播：计算概率对嵌入表示的梯度
    prob.backward()
    # 获取嵌入层的梯度：绝对值表示重要性大小，不受梯度方向影响
    # squeeze(0)移除batch维度，形状变为(seq_len, embedding_dim)
    gradients = embedded.grad.abs().squeeze(0)
    # 对每个token的所有嵌入维度求和，得到该token的总重要性
    # sum(dim=1)在嵌入维度上求和，形状变为(seq_len,)
    importance = gradients.sum(dim=1).detach().cpu().numpy()
    # 只保留实际文本长度内的重要性（去除padding部分）
    importance = importance[:length]

    # ========================================================================
    # Token解释：将token ID转换回文本
    # ========================================================================
    # 获取词汇表的ID到token映射
    tokens = vocab.idx_to_token
    # 将token ID转换为对应的token文本
    # 如果ID超出词汇表范围，使用"<unk>"表示未知token
    interpreted_tokens = [tokens[idx] if idx < len(tokens) else "<unk>" for idx in token_ids[:length]]

    # ========================================================================
    # 归一化：将重要性归一化为概率分布（总和为1）
    # ========================================================================
    # 计算重要性总和，加上小值避免除零
    total = importance.sum() + 1e-8
    # 归一化：每个token的重要性除以总和，得到相对重要性
    normalized = (importance / total).tolist()

    # 构建解释字典：包含原始文本、预测概率、token列表和归一化重要性
    explanation = {
        "text": text,  # 原始文本
        "probability": float(prob.item()),  # 预测概率（异常概率）
        "tokens": interpreted_tokens,  # token列表
        "importance": normalized,  # 每个token的归一化重要性
    }
    return explanation


# ============================================================================
# 显著性报告保存：为多个样本生成显著性分析并保存为JSON
# ============================================================================
def save_saliency_report(
    model: TextCNN,  # 训练好的模型
    vocab: Vocabulary,  # 词汇表
    samples: Iterable[str],  # 要分析的文本样本列表
    device: torch.device,  # 计算设备
    max_length: int,  # 最大序列长度
    output_path: Path,  # 输出文件路径
    top_k: int = 3,  # 保存的样本数量
) -> List[Dict[str, float]]:
    """为多个样本计算显著性并保存为JSON报告"""
    explanations: List[Dict[str, float]] = []  # 存储所有样本的解释
    # 遍历每个样本，计算其显著性
    for text in samples:
        # 计算当前样本的显著性解释
        explanations.append(compute_saliency(model, vocab, text, device, max_length))

    # 创建输出目录（如果不存在）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 将解释保存为JSON文件
    with output_path.open("w", encoding="utf-8") as f:
        # 只保存前top_k个样本的解释，ensure_ascii=False允许保存中文
        json.dump(explanations[:top_k], f, ensure_ascii=False, indent=2)
    return explanations  # 返回所有解释（不限于top_k）
