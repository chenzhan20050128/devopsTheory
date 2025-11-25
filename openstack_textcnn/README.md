# 基于轻量级 TextCNN 的 OpenStack 日志异常检测

本项目实现了《具体方案》中提出的轻量级 TextCNN 思路，并结合梯度显著性解释，面向 DevOps 场景输出可直接纳入流水线的指标与工件。

## ✨ 功能概览

- **日志预处理**：对 OpenStack 原始日志进行模板化归一（UUID/IP/数值占位），并直接以 `openstack_abnormal.log` 作为异常样本来源。
- **TextCNN 训练**：采用单层卷积 + 全局最大池化结构，保持模型轻量化，默认参数约 4 万级别。
- **模型评估**：输出 Accuracy / Precision / Recall / F1 / AUC，并生成 `metrics.json` 与 `test_predictions.npz` 供 CI 收集。
- **可解释性**：对最高风险样本计算梯度 × 输入贡献度，输出 `saliency_report.json` 展示关键信息片段。
- **DevOps 友好**：统一命令行入口，自动生成混淆矩阵图 (`confusion_matrix.png`)，方便接入 GitHub Actions / Jenkins 等流水线的制品归档步骤。
- **自适应阈值**：训练时自动网格搜索最优决策阈值，可选类平衡采样与手动 `pos_weight`，显著降低误报。

## � 数据说明

- 正常样本来自 `openstack_normal1.log` 与 `openstack_normal2.log`。
- 异常样本全部取自 `openstack_abnormal.log`；历史遗留的 `anomaly_labels.txt` 已不再参与标签构建。

## �📁 目录结构

```
openstack_textcnn/
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    ├── data.py          # 日志解析、归一化与 Dataset 封装
    ├── model.py         # TextCNN 主体，提供梯度解释接口
    ├── reporting.py     # 混淆矩阵绘制与显著性报告
    └── train.py         # 训练 / 评估 / 工件生成脚本
```

## ⚙️ 环境准备

```powershell
cd .\DL\openstack_textcnn
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Windows CPU 环境已验证；如使用 GPU，可去除运行命令中的 `--cpu` 开关。

## 🚀 快速训练

```powershell
python -m src.train `
  --data-dir ..\OpenStackData `
  --output-dir artifacts\latest_run `
  --epochs 5 `
  --batch-size 64 `
  --normal-downsample 0.3 `
  --balance-sampler `
  --threshold-mode optimize `
  --pos-weight 0 `
  --max-lines 40000 `
  --cpu
```

- `--max-lines` 与 `--normal-downsample` 可控制样本规模，便于在 CI 中快速运行。
- `--balance-sampler` 会以类别权重均衡采样，缓解异常样本稀缺带来的偏差。
- `--threshold-mode optimize` 会在验证集上搜索最优 F1 阈值；若想固定阈值，可改为 `fixed` 并搭配 `--threshold`。
- `--pos-weight 0` 表示自动估算 BCE 的正类权重；可根据需要显式传入其它数值。
- 训练完成后可在 `artifacts/` 目录看到：
  - `metrics.json`：包含数据集切分、验证曲线、测试指标。
  - `confusion_matrix.png`：模型在测试集上的表现可视化。
  - `saliency_report.json`：列出概率最高的异常样本及其关键 token。
  - `test_predictions.npz`：保存概率、预测结果，可用于后续一致性回放。

## 🧪 DevOps 集成建议

1. **CI/CD**：
   - Step 1：安装依赖（缓存虚拟环境或 wheel）。
   - Step 2：执行带采样的训练命令，产出工件。
   - Step 3：使用 `Upload Artifact` / `Archive` 插件收集 `artifacts/` 目录内容。
2. **质量门禁**：解析 `metrics.json` 中的 F1、AUC，设定阈值自动阻断低质量模型。
3. **Explainability Review**：在部署审批阶段，审阅 `saliency_report.json`，确认模型关注的关键信息合理。

## 🛠️ 关键实现要点

- 本实现采用 TextCNN（Kim 2014）架构，对照方案中的卷积核大小 `{3,4,5}`、嵌入维度 100、Dropout 0.5 等配置，可轻松扩展。
- 数据层面使用启发式模板化（UUID/IP/数值占位），在代码中清晰标注，便于针对具体场景替换为 Drain 等更强 parser。
- 显著性分析通过对嵌入层输出求梯度，得到 token 级贡献度，确保 **“能运行 + 可解释”** 两大必备要求。

## 📊 指标解读

`metrics.json` 中的关键字段说明：

- `dataset.total_records / train_size / val_size / test_size`：方便复盘本次实验覆盖范围。
- `train_history`：记录每个 epoch 的训练损失与验证指标，支持可视化趋势。
- `test`：最终测试集评估结果。
- `val_best_f1`：用于模型选择的最佳验证 F1，适合在流水线中作为基线比对。

## 🔀 可扩展方向

- 替换或叠加 Drain 在线模板抽取，进一步减少手工规则。
- 引入在线推理服务（FastAPI / Flask），将 `best_model.pt` 部署为实时接口。
- 结合 Prometheus / Grafana，将 `metrics.json` 指标推送到监控体系，实现全链路 DevOps 观测。

---

> 🚩 **说明**：代码内标注了所采用的方法及与方案的对应关系，如需切换算法（例如改用 LSTM-AE），建议复用同一数据与报告生成接口，保持流水线稳定。
