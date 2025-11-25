# 训练运行命令

## 快速开始（使用默认配置）

### Windows PowerShell
```powershell
cd "C:\Users\Administrator\Desktop\大三上学期\DevOps\DL"
python -m openstack_textcnn.src2.train
```

### Windows CMD
```cmd
cd /d "C:\Users\Administrator\Desktop\大三上学期\DevOps\DL"
python -m openstack_textcnn.src2.train
```

### 或者直接运行脚本
```powershell
# PowerShell
.\openstack_textcnn\src2\run_training.ps1
```

```cmd
# CMD
openstack_textcnn\src2\run_training.bat
```

## 完整命令（带所有参数）

### 使用默认配置（推荐）
```powershell
cd "C:\Users\Administrator\Desktop\大三上学期\DevOps\DL"
python -m openstack_textcnn.src2.train
```

### 自定义输出目录
```powershell
cd "C:\Users\Administrator\Desktop\大三上学期\DevOps\DL"
python -m openstack_textcnn.src2.train --output-dir "openstack_textcnn/artifacts_forecast_v2"
```

### 使用CPU训练（如果没有GPU）
```powershell
cd "C:\Users\Administrator\Desktop\大三上学期\DevOps\DL"
python -m openstack_textcnn.src2.train --device cpu
```

### 快速测试（减少数据量和训练轮数）
```powershell
cd "C:\Users\Administrator\Desktop\大三上学期\DevOps\DL"
python -m openstack_textcnn.src2.train --max-lines-per-file 1000 --epochs 5 --output-dir "openstack_textcnn/artifacts_forecast_test"
```

### 完整自定义参数
```powershell
cd "C:\Users\Administrator\Desktop\大三上学期\DevOps\DL"
python -m openstack_textcnn.src2.train `
    --device cuda `
    --epochs 20 `
    --batch-size 32 `
    --lr 0.0005 `
    --output-dir "openstack_textcnn/artifacts_forecast_custom" `
    --window-size 8 `
    --window-stride 4 `
    --sequence-length 12 `
    --alert-horizon 3
```

## 参数说明

- `--device`: 设备类型，`cuda` 或 `cpu`（默认：`cuda`）
- `--epochs`: 训练轮数（默认：`20`）
- `--batch-size`: 批次大小（默认：`32`）
- `--lr`: 学习率（默认：`0.0005`）
- `--output-dir`: 输出目录（默认：`openstack_textcnn/artifacts_forecast`）
- `--max-lines-per-file`: 每个文件最大行数，用于快速测试（默认：`None`，使用全部数据）
- `--window-size`: 窗口大小（默认：`8`）
- `--window-stride`: 窗口步长（默认：`4`）
- `--sequence-length`: 序列长度（默认：`12`）
- `--alert-horizon`: 预警时间范围（默认：`3`）
- `--normal-downsample`: 正常样本下采样比例（默认：`1.0`，不下采样）

## 输出文件

训练完成后，会在输出目录生成：
- `forecast_model.pt`: 训练好的模型
- `forecast_metrics.json`: 训练指标和评估结果

## 检查训练结果

训练过程中会输出：
1. 数据集统计信息（样本数、正负样本比例等）
2. 每个epoch的训练和验证指标
3. 最终测试集评估结果

训练完成后，查看 `forecast_metrics.json` 文件获取详细指标。

