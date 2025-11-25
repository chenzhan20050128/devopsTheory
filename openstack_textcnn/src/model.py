"""TextCNN 模型定义：轻量化卷积结构用于日志异常检测。

架构遵循 Kim (2014) TextCNN，并结合《具体方案》中提出的轻量级配置：
- 嵌入维度 100，卷积核尺寸 {3,4,5}；
- 全局最大池化提取局部模板特征；
- Dropout + 全连接压缩后进入 Sigmoid 输出。

提供 `forward_with_embeddings` 接口以支持梯度显著性解释。
"""

# 导入类型注解模块，用于函数参数和返回值的类型提示
from typing import Iterable, List, Optional

# 导入PyTorch深度学习框架，用于构建神经网络
import torch
# 导入PyTorch神经网络模块，包含各种层和激活函数
from torch import nn


# ============================================================================
# TextCNN模型类：用于日志异常检测的轻量级卷积神经网络
# ============================================================================
class TextCNN(nn.Module):
    """Lightweight TextCNN model tailored for log anomaly detection."""  # 轻量级TextCNN模型，专用于日志异常检测

    # ========================================================================
    # 模型初始化：构建TextCNN的各个组件
    # ========================================================================
    def __init__(
        self,
        vocab_size: int,  # 词汇表大小（输入token的数量）
        embedding_dim: int = 100,  # 词嵌入维度，默认100维
        num_filters: int = 64,  # 每个卷积层的滤波器数量
        kernel_sizes: Iterable[int] = (3, 4, 5),  # 卷积核尺寸的可迭代对象，默认3,4,5
        dropout: float = 0.5,  # Dropout比率，防止过拟合
        hidden_dim: int = 128,  # 全连接层隐藏单元维度
        numeric_feature_dim: int = 0,  # 数值特征维度，0表示不使用数值特征
        numeric_hidden_dim: int = 32,  # 数值特征编码器的隐藏层维度
    ) -> None:
        # 调用父类nn.Module的初始化方法，确保PyTorch模块正确初始化
        super().__init__()

        # ====================================================================
        # 词嵌入层：将离散的token索引映射为连续的向量表示
        # ====================================================================
        # 定义词嵌入层，vocab_size为词汇表大小，embedding_dim为嵌入维度
        # padding_idx=0表示索引0用于填充，填充位置的梯度不会被更新
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # 保存数值特征维度，用于后续判断是否使用数值特征
        self.numeric_feature_dim = numeric_feature_dim

        # ====================================================================
        # 多尺度卷积层：使用不同尺寸的卷积核捕获不同长度的文本模式
        # ====================================================================
        # 定义卷积层模块列表，包含不同尺寸的卷积核（如3,4,5）
        # ModuleList允许动态创建多个卷积层，每个层处理不同长度的n-gram特征
        self.convs = nn.ModuleList(
            [
                # 一维卷积层：在序列维度上进行卷积操作
                nn.Conv1d(
                    in_channels=embedding_dim,  # 输入通道数等于嵌入维度（每个token的向量维度）
                    out_channels=num_filters,  # 输出通道数等于滤波器数量（每个卷积核产生的特征图数量）
                    kernel_size=k,  # 卷积核大小（捕获k个连续token的模式）
                    padding=0,  # 不进行填充，输出序列长度会缩短
                )
                for k in kernel_sizes  # 为每个卷积核尺寸创建独立的卷积层
            ]
        )
        # ReLU激活函数：引入非线性，使模型能够学习复杂模式
        self.activation = nn.ReLU()
        # Dropout层：随机将部分神经元输出置零，防止过拟合
        self.dropout = nn.Dropout(dropout)
        # 计算卷积层输出总维度：每个卷积核产生num_filters个特征，多个卷积核拼接
        conv_output_dim = num_filters * len(tuple(kernel_sizes))

        # ====================================================================
        # 数值特征编码器：处理额外的数值统计特征（如日志中的数字统计）
        # ====================================================================
        if numeric_feature_dim > 0:  # 如果使用数值特征
            # 定义数值特征编码器序列：将原始数值特征转换为隐藏表示
            self.numeric_encoder = nn.Sequential(
                nn.LayerNorm(numeric_feature_dim),  # 层归一化：稳定训练，加速收敛
                nn.Linear(numeric_feature_dim, numeric_hidden_dim),  # 线性变换：降维到隐藏维度
                nn.ReLU(),  # ReLU激活：引入非线性
                nn.Dropout(dropout),  # Dropout：防止过拟合
            )
            # 全连接层输入维度 = 文本卷积特征维度 + 数值特征编码维度
            fc_input_dim = conv_output_dim + numeric_hidden_dim
        else:  # 如果不使用数值特征
            self.numeric_encoder = None  # 数值编码器设为None
            fc_input_dim = conv_output_dim  # 全连接层输入维度仅包含卷积输出维度

        # ====================================================================
        # 全连接层和分类器：融合特征并进行最终分类
        # ====================================================================
        # 全连接层：将拼接后的特征进行融合和降维
        self.fc = nn.Linear(fc_input_dim, hidden_dim)
        # 分类器层：输出1维logits（二分类任务，后续通过sigmoid得到概率）
        self.classifier = nn.Linear(hidden_dim, 1)

        # 调用权重初始化方法，确保模型参数在合理范围内
        self._init_weights()

    # ========================================================================
    # 权重初始化：使用合适的初始化策略确保训练稳定性
    # ========================================================================
    def _init_weights(self) -> None:
        # Xavier均匀初始化词嵌入权重：适合tanh/sigmoid激活，保证梯度稳定
        nn.init.xavier_uniform_(self.embedding.weight)
        # 遍历所有卷积层进行初始化
        for conv in self.convs:
            # Kaiming均匀初始化卷积权重：专门针对ReLU激活函数优化
            # 能够保持前向传播时激活值的方差，避免梯度消失或爆炸
            nn.init.kaiming_uniform_(conv.weight, nonlinearity="relu")
            # 将卷积偏置初始化为0：标准做法
            nn.init.zeros_(conv.bias)
        # 如果存在数值编码器，初始化其线性层
        if self.numeric_encoder is not None:
            for layer in self.numeric_encoder:
                if isinstance(layer, nn.Linear):  # 只初始化线性层
                    nn.init.xavier_uniform_(layer.weight)  # Xavier初始化权重
                    nn.init.zeros_(layer.bias)  # 偏置初始化为0
        # 初始化全连接层和分类器
        nn.init.xavier_uniform_(self.fc.weight)  # Xavier初始化全连接层权重
        nn.init.zeros_(self.fc.bias)  # 全连接层偏置初始化为0
        nn.init.xavier_uniform_(self.classifier.weight)  # Xavier初始化分类器权重
        nn.init.zeros_(self.classifier.bias)  # 分类器偏置初始化为0

    # ========================================================================
    # 内部前向传播方法：从嵌入表示开始处理（供其他方法复用）
    # ========================================================================
    def _forward_from_embedded(
        self,
        embedded: torch.Tensor,  # 输入嵌入张量，形状为(batch, embedding_dim, seq_len)
        numeric_features: Optional[torch.Tensor] = None,  # 可选的数值特征张量
    ) -> torch.Tensor:
        # ====================================================================
        # 多尺度卷积特征提取：使用不同尺寸卷积核捕获文本模式
        # ====================================================================
        conv_outputs: List[torch.Tensor] = []  # 初始化卷积输出列表
        # 遍历每个卷积层（不同尺寸的卷积核）
        for conv in self.convs:
            # 卷积操作：在序列维度上滑动卷积核，提取局部特征
            # 输出形状：(batch, num_filters, new_seq_len)，new_seq_len = seq_len - kernel_size + 1
            x = self.activation(conv(embedded))  # 应用ReLU激活函数，引入非线性
            # 全局最大池化：沿序列维度取最大值，提取最重要的特征
            # dim=2表示在序列维度上操作，.values获取最大值（忽略索引）
            # 输出形状：(batch, num_filters)，每个滤波器保留一个最重要的激活值
            x = torch.max(x, dim=2).values
            conv_outputs.append(x)  # 将池化结果添加到列表

        # ====================================================================
        # 特征拼接：融合不同卷积核提取的特征
        # ====================================================================
        # 在特征维度（dim=1）拼接不同卷积核的输出
        # 输出形状：(batch, num_filters * len(kernel_sizes))
        concatenated = torch.cat(conv_outputs, dim=1)
        # 对拼接特征应用Dropout，防止过拟合
        feature_list = [self.dropout(concatenated)]

        # ====================================================================
        # 数值特征融合：如果使用数值特征，将其编码后与文本特征拼接
        # ====================================================================
        if self.numeric_encoder is not None:  # 如果使用数值特征
            if numeric_features is None:  # 如果未提供数值特征
                # 创建全零数值特征张量作为占位符
                numeric_features = torch.zeros(
                    concatenated.size(0),  # batch大小，从拼接特征中获取
                    self.numeric_feature_dim,  # 数值特征维度
                    device=concatenated.device,  # 设备保持一致（CPU/GPU）
                    dtype=concatenated.dtype,  # 数据类型保持一致
                )
            # 通过数值编码器处理数值特征：归一化、线性变换、激活、Dropout
            numeric_repr = self.numeric_encoder(numeric_features)
            feature_list.append(numeric_repr)  # 将数值特征表示添加到特征列表

        # ====================================================================
        # 特征融合与分类：将文本和数值特征融合后进行分类
        # ====================================================================
        # 在特征维度拼接文本特征和数值特征
        fused_features = torch.cat(feature_list, dim=1)
        # 全连接层变换：将融合特征映射到隐藏维度
        hidden = self.activation(self.fc(fused_features))  # 应用ReLU激活
        hidden = self.dropout(hidden)  # 应用Dropout，防止过拟合
        # 分类器：将隐藏表示映射到1维logits（未归一化的分数）
        logits = self.classifier(hidden)
        # 压缩第1维度：从[batch, 1]变为[batch]，用于二分类任务
        return logits.squeeze(1)

    # ========================================================================
    # 标准前向传播方法：从token索引开始处理（常规使用）
    # ========================================================================
    def forward(
        self,
        inputs: torch.Tensor,  # 输入token索引张量，形状(batch, seq_len)
        numeric_features: Optional[torch.Tensor] = None,  # 可选的数值特征
    ) -> torch.Tensor:
        # 通过嵌入层将token索引转换为向量表示
        # 输入形状：(batch, seq_len)，输出形状：(batch, seq_len, embedding_dim)
        embedded = self.embedding(inputs)
        # 转置维度以适应卷积层输入要求
        # Conv1d期望输入形状为(batch, channels, seq_len)
        # 转置后形状：(batch, embedding_dim, seq_len)
        embedded = embedded.transpose(1, 2)
        # 调用内部前向传播方法，完成后续处理
        return self._forward_from_embedded(embedded, numeric_features)

    # ========================================================================
    # 从嵌入开始的前向传播方法：用于梯度显著性解释
    # ========================================================================
    def forward_with_embeddings(
        self,
        embedded: torch.Tensor,  # 直接输入嵌入张量（跳过嵌入层，用于梯度计算）
        numeric_features: Optional[torch.Tensor] = None,  # 可选的数值特征
    ) -> torch.Tensor:
        # 直接调用内部前向传播方法，跳过嵌入层
        # 这个方法主要用于计算梯度显著性，需要保留嵌入层的梯度
        return self._forward_from_embedded(embedded, numeric_features)