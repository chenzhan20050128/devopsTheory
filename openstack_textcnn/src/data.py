"""OpenStack 日志预处理与数据集封装模块.

核心能力：
- 模板化: 统一替换 UUID / 请求 ID / IP / 数值占位, 降噪.
- 数值信息: 提取数值统计量作为附加特征, 供模型多模态融合.
- 滑动窗口: 支持窗口大小 / 步长配置, 聚合上下文.
- Dataset: 暴露词表构建与 PyTorch 兼容的数据集接口.
"""

# 导入未来版本的注解特性，允许使用更灵活的类型注解
from __future__ import annotations

# 导入正则表达式模块，用于文本模式匹配和替换
import re
# 导入数据类装饰器，用于创建简单的数据容器类
from dataclasses import dataclass
# 导入路径处理模块
from pathlib import Path
# 导入类型注解
from typing import Iterable, List, Optional, Sequence, Set, Tuple

# 导入NumPy数值计算库
import numpy as np
# 导入Pandas数据处理库
import pandas as pd
# 导入sklearn的数据划分函数
from sklearn.model_selection import train_test_split

# ============================================================================
# 正则表达式模式：用于识别和替换日志中的特定模式
# ============================================================================
# UUID模式：匹配标准UUID格式（8-4-4-4-12个十六进制字符）
UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)
# 请求ID模式：匹配OpenStack请求ID格式（req-后跟十六进制字符和连字符）
REQUEST_ID_PATTERN = re.compile(r"req-[0-9a-f-]+", re.IGNORECASE)
# IP地址模式：匹配IPv4地址格式（四个0-255的数字，用点分隔）
IP_PATTERN = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
# 数字模式：匹配整数或浮点数（可选小数点）
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
# 空白字符模式：匹配一个或多个连续空白字符（空格、制表符等）
WHITESPACE_PATTERN = re.compile(r"\s+")

# ============================================================================
# 常量定义
# ============================================================================
# 数值特征维度：提取5个统计量（数量、均值、最大值、最小值、标准差）
NUMERIC_FEATURE_DIM = 5
# 异常关键词：用于识别异常日志的关键词列表
ABNORMAL_KEYWORDS = ("error", "failed", "exception", "traceback", "critical", "panic", "fatal")


# ============================================================================
# 数据集划分数据类：存储训练、验证、测试集的DataFrame
# ============================================================================
@dataclass(frozen=True)  # frozen=True使实例不可变，确保数据安全
class DatasetSplits:
    """训练 / 验证 / 测试数据划分."""

    train: pd.DataFrame  # 训练集DataFrame
    val: pd.DataFrame  # 验证集DataFrame
    test: pd.DataFrame  # 测试集DataFrame

# ============================================================================
# 辅助函数：从日志行中提取信息
# ============================================================================
def _extract_instance_id(line: str) -> Optional[str]:
    """从日志行中提取实例ID（UUID格式）"""
    # 使用正则表达式搜索实例ID：匹配"instance: "后的36个字符（UUID格式）
    match = re.search(r"instance: ([0-9a-f-]{36})", line, re.IGNORECASE)
    if match:
        # 如果找到匹配，返回小写的实例ID
        return match.group(1).lower()
    return None  # 如果未找到，返回None


def _extract_message(line: str) -> str:
    """从日志行中提取消息部分（去除时间戳和日志级别等前缀）"""
    # 如果日志行包含"]"，说明有结构化前缀（如时间戳、日志级别）
    if "]" in line:
        # 在第一个"]"后分割，取后面的部分作为消息
        message = line.split("]", maxsplit=1)[-1]
    else:
        # 如果没有"]"，整行都是消息
        message = line
    # 去除首尾空白字符并返回
    return message.strip()


def _extract_numeric_features(message: str) -> np.ndarray:
    """从消息中提取数值特征（统计量）"""
    # 使用正则表达式查找所有数字（整数或浮点数）
    numbers = [float(match) for match in NUMBER_PATTERN.findall(message)]
    # 如果没有找到数字，返回全零特征向量
    if not numbers:
        return np.zeros(NUMERIC_FEATURE_DIM, dtype=np.float32)

    # 将数字列表转换为NumPy数组
    arr = np.array(numbers, dtype=np.float32)
    # 计算5个统计特征：数量、均值、最大值、最小值、标准差
    return np.array(
        [
            float(arr.size),  # 数字的数量
            float(arr.mean()),  # 数字的均值
            float(arr.max()),  # 数字的最大值
            float(arr.min()),  # 数字的最小值
            float(arr.std(ddof=0)),  # 数字的标准差（ddof=0表示总体标准差）
        ],
        dtype=np.float32,
    )


# ============================================================================
# 消息归一化：将日志消息标准化为模型可处理的格式
# ============================================================================
def normalize_message(message: str) -> str:
    """将日志消息归一化为标准格式：替换特定模式为占位符，去除特殊字符"""
    # 转换为小写：统一大小写，减少词汇表大小
    message = message.lower()
    # 替换UUID为占位符：统一所有UUID为"<uuid>"
    message = UUID_PATTERN.sub(" <uuid> ", message)
    # 替换请求ID为占位符：统一所有请求ID为"<req>"
    message = REQUEST_ID_PATTERN.sub(" <req> ", message)
    # 替换IP地址为占位符：统一所有IP地址为"<ip>"
    message = IP_PATTERN.sub(" <ip> ", message)
    # 替换数字为占位符：统一所有数字为"<num>"
    message = NUMBER_PATTERN.sub(" <num> ", message)
    # 去除所有非字母、非占位符、非空白字符：只保留字母、占位符和空白
    message = re.sub(r"[^a-z<>\s]", " ", message)
    # 将多个连续空白字符替换为单个空格：规范化空白
    message = WHITESPACE_PATTERN.sub(" ", message)
    # 去除首尾空白并返回
    return message.strip()


# ============================================================================
# 异常实例ID加载：从文件中读取标记为异常的实例ID
# ============================================================================
def load_anomaly_instance_ids(file_path: Path) -> Set[str]:
    """从文件中加载异常实例ID集合"""
    # 如果文件不存在，返回空集合
    if not file_path.exists():
        return set()

    anomaly_ids: Set[str] = set()  # 初始化异常ID集合
    # 打开文件并逐行读取
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # 在每行中搜索UUID模式
            match = UUID_PATTERN.search(line)
            if match:
                # 如果找到UUID，添加到集合中（小写）
                anomaly_ids.add(match.group(0).lower())
    return anomaly_ids  # 返回异常ID集合


# ============================================================================
# 单文件加载：从单个日志文件中加载和预处理数据
# ============================================================================
def _load_single_file(
    file_path: Path,  # 日志文件路径
    label: int,  # 文件的基础标签（0=正常，1=异常）
    max_lines: Optional[int],  # 最大读取行数（None表示全部）
    downsample_ratio: float,  # 正常样本下采样比例（用于类别平衡）
    anomaly_instances: Optional[Set[str]] = None,  # 异常实例ID集合（用于精细标注）
) -> List[Tuple[str, int, Optional[str], str, np.ndarray]]:
    """从单个日志文件中加载数据，返回(文本, 标签, 实例ID, 文件名, 数值特征)元组列表"""
    rows: List[Tuple[str, int, Optional[str], str, np.ndarray]] = []  # 存储处理后的数据行
    tracked_anomaly_requests: Set[str] = set()  # 跟踪已发现的异常请求ID
    use_anomaly_labels = bool(anomaly_instances)  # 是否使用异常实例标注

    # 如果未提供异常实例集合，初始化为空集合
    if anomaly_instances is None:
        anomaly_instances = set()

    # 打开文件并逐行处理
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, raw_line in enumerate(f):
            # 如果设置了最大行数限制，达到限制后停止
            if max_lines is not None and idx >= max_lines:
                break

            # 提取日志消息部分（去除时间戳等前缀）
            message = _extract_message(raw_line)
            if not message:  # 如果消息为空，跳过
                continue

            # 提取实例ID（如果存在）
            instance_id = _extract_instance_id(raw_line)
            if instance_id:
                instance_id = instance_id.lower()  # 转换为小写
            # 提取数值特征（统计量）
            numeric_features = _extract_numeric_features(message)
            # 归一化消息（替换UUID、IP等为占位符）
            normalized = normalize_message(message)
            if not normalized:  # 如果归一化后为空，跳过
                continue

            # ====================================================================
            # 标签确定：根据异常实例标注和关键词确定有效标签
            # ====================================================================
            effective_label = label  # 初始化为文件的基础标签
            if use_anomaly_labels:  # 如果使用异常实例标注
                # 提取请求ID列表（小写）
                request_ids = [req.lower() for req in REQUEST_ID_PATTERN.findall(raw_line)]
                # 如果实例ID在异常实例集合中，标记为异常
                if instance_id and instance_id in anomaly_instances:
                    effective_label = 1
                # 如果基础标签是异常，且请求ID在已跟踪的异常请求中，标记为异常
                elif label == 1 and request_ids and any(req in tracked_anomaly_requests for req in request_ids):
                    effective_label = 1
                # 如果基础标签是异常，且消息包含异常关键词，标记为异常
                elif label == 1 and any(keyword in normalized for keyword in ABNORMAL_KEYWORDS):
                    effective_label = 1
                else:
                    effective_label = 0  # 否则标记为正常

                # 如果当前行被标记为异常且有请求ID，将请求ID添加到跟踪集合
                if effective_label == 1 and request_ids:
                    tracked_anomaly_requests.update(request_ids)

            # ====================================================================
            # 下采样：对正常样本进行随机下采样，处理类别不平衡
            # ====================================================================
            # 如果是正常样本且下采样比例<1.0，随机决定是否保留
            if effective_label == 0 and downsample_ratio < 1.0 and np.random.rand() > downsample_ratio:
                continue  # 随机丢弃，不添加到结果中

            # 将处理后的数据添加到结果列表
            rows.append((normalized, effective_label, instance_id, file_path.name, numeric_features))
    # 注意：以下代码似乎是重复的，但保留原样
    rows: List[Tuple[str, int, Optional[str], str, np.ndarray]] = []
    anomaly_instances: Set[str] = getattr(_load_single_file, "anomaly_instances", set())
    anomaly_request_ids: Set[str] = set()

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, raw_line in enumerate(f):
            if max_lines is not None and idx >= max_lines:
                break

            message = _extract_message(raw_line)
            if not message:
                continue

            instance_id = _extract_instance_id(raw_line)
            numeric_features = _extract_numeric_features(message)
            normalized = normalize_message(message)
            if not normalized:
                continue

            effective_label = label
            if anomaly_instances:
                request_ids = [req.lower() for req in REQUEST_ID_PATTERN.findall(raw_line)]
                if instance_id and instance_id in anomaly_instances:
                    effective_label = 1
                elif label == 1 and request_ids and any(req in anomaly_request_ids for req in request_ids):
                    effective_label = 1
                elif label == 1 and any(keyword in normalized for keyword in ABNORMAL_KEYWORDS):
                    effective_label = 1
                else:
                    effective_label = 0

                if effective_label == 1 and request_ids:
                    anomaly_request_ids.update(request_ids)

            if effective_label == 0 and downsample_ratio < 1.0 and np.random.rand() > downsample_ratio:
                continue

            rows.append((normalized, effective_label, instance_id, file_path.name, numeric_features))
    return rows


# ============================================================================
# 滑动窗口应用：将连续的日志行聚合为窗口，捕获上下文信息
# ============================================================================
def apply_sliding_window(df: pd.DataFrame, window_size: int, window_stride: int) -> pd.DataFrame:
    """使用滑动窗口将数据聚合为窗口，每个窗口包含连续的window_size行"""
    # 确保窗口大小大于1（否则没有聚合意义）
    assert window_size > 1

    records: List[Tuple[str, int, Optional[str], str, np.ndarray]] = []  # 存储窗口数据
    columns = ["text", "label", "instance_id", "source", "features"]  # DataFrame列名

    # 按源文件分组处理：每个文件独立应用滑动窗口
    for source, group in df.groupby("source", sort=False):
        # 提取各组件的列表
        texts = group["text"].tolist()  # 文本列表
        features = np.stack(group["features"].to_numpy())  # 数值特征数组（堆叠）
        labels = group["label"].tolist()  # 标签列表
        instances = group["instance_id"].tolist()  # 实例ID列表

        # 应用滑动窗口：从0开始，步长为window_stride
        for start in range(0, len(group) - window_size + 1, window_stride):
            end = start + window_size  # 窗口结束位置
            # 将窗口内的文本用空格连接
            window_text = " ".join(texts[start:end])
            # 窗口标签：如果窗口内任何一行是异常，则窗口为异常
            window_label = int(any(labels[start:end]))
            # 窗口数值特征：取窗口内特征的均值
            window_features = features[start:end].mean(axis=0)
            # 窗口实例ID：使用窗口最后一个元素的实例ID
            window_instance = instances[end - 1]
            # 将窗口数据添加到记录列表
            records.append((window_text, window_label, window_instance, source, window_features))

    # 创建窗口DataFrame
    window_df = pd.DataFrame(records, columns=columns)
    # 重置索引（去除原始索引）
    window_df.reset_index(drop=True, inplace=True)
    return window_df


# ============================================================================
# 数据集加载：从多个日志文件中加载数据并构建DataFrame
# ============================================================================
def load_dataset(
    data_dir: Path,  # 数据目录路径
    max_lines_per_file: Optional[int] = None,  # 每个文件最大读取行数
    normal_downsample: float = 1.0,  # 正常样本下采样比例
    deduplicate: bool = False,  # 是否去重
    window_size: int = 1,  # 滑动窗口大小（1表示不使用窗口）
    window_stride: int = 1,  # 滑动窗口步长
) -> pd.DataFrame:
    """加载OpenStack日志数据集，返回包含文本、标签、特征等的DataFrame"""
    # 定义要加载的文件列表及其基础标签
    files = [
        (data_dir / "openstack_normal1.log", 0),  # 正常日志文件1，标签为0
        (data_dir / "openstack_normal2.log", 0),  # 正常日志文件2，标签为0
        (data_dir / "openstack_abnormal.log", 1),  # 异常日志文件，标签为1
    ]

    # 加载异常实例ID集合（如果存在标注文件）
    anomaly_instances = load_anomaly_instance_ids(data_dir / "anomaly_labels.txt")
    dataset_rows: List[Tuple[str, int, Optional[str], str, np.ndarray]] = []  # 存储所有数据行
    # 遍历每个文件，加载数据
    for file_path, label in files:
        rows = _load_single_file(
            file_path=file_path,  # 文件路径
            label=label,  # 基础标签
            max_lines=max_lines_per_file,  # 最大行数
            downsample_ratio=normal_downsample if label == 0 else 1.0,  # 只有正常样本才下采样
            anomaly_instances=anomaly_instances,  # 异常实例集合
        )
        dataset_rows.extend(rows)  # 将当前文件的数据添加到总列表

    # 创建DataFrame：包含文本、标签、实例ID、源文件名、数值特征
    df = pd.DataFrame(dataset_rows, columns=["text", "label", "instance_id", "source", "features"])

    # 如果启用去重，去除重复的(文本, 标签)组合
    if deduplicate:
        df = df.drop_duplicates(subset=["text", "label"], keep="first")  # 保留第一个重复项

    # 重置索引
    df.reset_index(drop=True, inplace=True)

    # 如果窗口大小>1，应用滑动窗口聚合
    if window_size > 1:
        df = apply_sliding_window(df, window_size=window_size, window_stride=window_stride)

    return df


# ============================================================================
# 数据集划分：将数据集划分为训练集、验证集、测试集
# ============================================================================
def split_dataset(
    df: pd.DataFrame,  # 完整数据集
    train_ratio: float = 0.7,  # 训练集比例
    val_ratio: float = 0.15,  # 验证集比例（剩余部分为测试集）
    random_state: int = 42,  # 随机种子
) -> DatasetSplits:
    """将数据集划分为训练、验证、测试集，保持类别分布一致（分层抽样）"""
    # 第一次划分：分离训练集和临时集（验证+测试）
    # stratify=df["label"]确保训练集和临时集的类别分布与原始数据一致
    train_df, temp_df = train_test_split(
        df,
        test_size=1 - train_ratio,  # 临时集大小 = 1 - 训练集比例
        random_state=random_state,  # 随机种子
        stratify=df["label"],  # 分层抽样，保持类别比例
    )
    # 计算验证集在临时集中的相对比例
    # 例如：如果val_ratio=0.15，train_ratio=0.7，则relative_val = 0.15/(1-0.7) = 0.5
    relative_val = val_ratio / (1 - train_ratio)

    # 第二次划分：从临时集中分离验证集和测试集
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_val,  # 测试集在临时集中的比例
        random_state=random_state,  # 使用相同随机种子
        stratify=temp_df["label"],  # 继续分层抽样
    )

    # 返回数据集划分对象，重置索引
    return DatasetSplits(
        train=train_df.reset_index(drop=True),  # 训练集
        val=val_df.reset_index(drop=True),  # 验证集
        test=test_df.reset_index(drop=True),  # 测试集
    )


# ============================================================================
# 词汇表类：管理token到ID的映射，用于文本编码
# ============================================================================
class Vocabulary:
    """词汇表：将文本token映射为整数ID"""
    
    def __init__(self, min_freq: int = 1, max_size: Optional[int] = None):
        """初始化词汇表"""
        self.min_freq = min_freq  # 最小词频：低于此频率的词将被忽略
        self.max_size = max_size  # 最大词汇表大小：None表示不限制
        # 初始化特殊token：padding和unknown
        self.token_to_idx = {"<pad>": 0, "<unk>": 1}  # token到ID的映射
        self.idx_to_token = ["<pad>", "<unk>"]  # ID到token的映射

    def build(self, texts: Iterable[str]) -> None:
        """从文本集合中构建词汇表"""
        # 统计每个token的出现频率
        freq: dict[str, int] = {}
        for text in texts:  # 遍历所有文本
            for token in text.split():  # 按空格分割文本为token
                freq[token] = freq.get(token, 0) + 1  # 累加频率

        # 按频率降序、token升序排序：频率高的优先，相同频率按字母序
        sorted_tokens = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
        # 遍历排序后的token，添加到词汇表
        for token, count in sorted_tokens:
            # 如果频率低于最小频率，跳过
            if count < self.min_freq:
                continue
            # 如果token已在词汇表中，跳过（避免重复）
            if token in self.token_to_idx:
                continue
            # 如果达到最大词汇表大小，停止添加
            if self.max_size is not None and len(self.idx_to_token) >= self.max_size:
                break
            # 添加token到词汇表：ID为当前词汇表大小
            self.token_to_idx[token] = len(self.idx_to_token)
            self.idx_to_token.append(token)

    def encode(self, text: str) -> List[int]:
        """将文本编码为token ID列表"""
        # 将文本分割为token，查找每个token的ID
        # 如果token不在词汇表中，使用<unk>的ID（1）
        return [self.token_to_idx.get(token, 1) for token in text.split()]

    def __len__(self) -> int:
        """返回词汇表大小"""
        return len(self.idx_to_token)


# ============================================================================
# 日志数据集类：PyTorch兼容的数据集，用于DataLoader
# ============================================================================
class LogDataset:
    """PyTorch数据集类：将DataFrame转换为模型可用的格式"""
    
    def __init__(self, df: pd.DataFrame, vocab: Vocabulary, max_length: int = 200):
        """初始化数据集"""
        self.df = df.reset_index(drop=True)  # 重置DataFrame索引
        self.vocab = vocab  # 词汇表对象
        self.max_length = max_length  # 最大序列长度

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """获取指定索引的数据样本"""
        # 获取指定行的数据
        row = self.df.iloc[idx]
        # 使用词汇表将文本编码为token ID列表
        token_ids = self.vocab.encode(row["text"])
        # 如果序列长度超过最大长度，进行截断
        if len(token_ids) >= self.max_length:
            token_ids = token_ids[: self.max_length]
        else:  # 如果序列长度不足，进行填充
            # 使用0（padding token）填充到max_length
            token_ids = token_ids + [0] * (self.max_length - len(token_ids))

        # 将数值特征转换为NumPy数组
        features = np.asarray(row["features"], dtype=np.float32)
        # 返回(token IDs, 数值特征, 标签)元组
        return np.array(token_ids, dtype=np.int64), features, int(row["label"])

    @staticmethod
    def collate(batch: Sequence[Tuple[np.ndarray, np.ndarray, int]]):
        """批处理函数：将多个样本组合成一个批次"""
        # 堆叠token IDs：从列表堆叠为(batch_size, max_length)数组
        token_batch = np.stack([item[0] for item in batch], axis=0)
        # 堆叠数值特征：从列表堆叠为(batch_size, feature_dim)数组
        feature_batch = np.stack([item[1] for item in batch], axis=0)
        # 堆叠标签：从列表转换为(batch_size,)数组
        label_batch = np.array([item[2] for item in batch], dtype=np.float32)

        # 导入torch（在函数内部导入，避免循环依赖）
        import torch

        # 将NumPy数组转换为PyTorch张量
        inputs = torch.as_tensor(token_batch, dtype=torch.long)  # token IDs为长整型
        numeric_features = torch.as_tensor(feature_batch, dtype=torch.float32)  # 数值特征为浮点型
        labels = torch.as_tensor(label_batch, dtype=torch.float32)  # 标签为浮点型（用于BCE损失）
        return inputs, numeric_features, labels
