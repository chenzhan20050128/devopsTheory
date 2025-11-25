"""诊断脚本：检查测试集评估结果的合理性"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

def analyze_metrics(metrics_file: Path):
    """分析指标文件，检查是否存在异常"""
    with open(metrics_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_metrics = data.get('test_metrics', {})
    
    print("=" * 60)
    print("测试集指标分析")
    print("=" * 60)
    
    # 检查损失
    loss = test_metrics.get('loss', 0)
    print(f"\n1. 损失值: {loss:.6f}")
    if loss == 0.0:
        print("   [WARNING] 损失为0，这很不正常！")
        print("   可能原因:")
        print("   - 损失计算有bug")
        print("   - 测试集为空或样本数计算错误")
        print("   - 模型预测过于完美（不太可能）")
    elif loss < 0.0001:
        print("   [WARNING] 损失非常小，可能存在问题")
    
    # 检查Recall和Precision
    recall = test_metrics.get('recall@top', 0)
    precision = test_metrics.get('precision@top', 0)
    print(f"\n2. Recall@Top (20%): {recall:.6f}")
    print(f"   Precision@Top (20%): {precision:.6f}")
    
    if precision == 1.0 and recall < 0.5:
        print("   [WARNING] Precision=1.0但Recall较低，可能的原因:")
        print("   - 测试集中正样本数量很少")
        print("   - 模型预测的top 20%样本中，正样本数量正好等于真实正样本的20%")
        print("   - 数据分布极不平衡")
    
    # 检查AUC
    pr_auc = test_metrics.get('pr_curve', {}).get('auc', 0)
    roc_auc = test_metrics.get('roc_curve', {}).get('auc', 0)
    print(f"\n3. PR-AUC: {pr_auc:.6f}")
    print(f"   ROC-AUC: {roc_auc:.6f}")
    
    if pr_auc == 1.0 and roc_auc == 1.0:
        print("   [WARNING] AUC都是1.0，这通常意味着:")
        print("   - 模型完美分离了正负样本（不太可能）")
        print("   - 测试集中只有一个类别（更可能）")
        print("   - 数据或指标计算有问题")
    
    # 检查回归指标
    reg_mae = test_metrics.get('reg_mae', 0)
    reg_rmse = test_metrics.get('reg_rmse', 0)
    print(f"\n4. 回归 MAE: {reg_mae:.6f}")
    print(f"   回归 RMSE: {reg_rmse:.6f}")
    
    # 检查验证历史
    val_history = data.get('val_history', [])
    if val_history:
        last_val = val_history[-1]
        print(f"\n5. 最后一个验证epoch:")
        print(f"   损失: {last_val.get('loss', 0):.6f}")
        print(f"   Recall@Top: {last_val.get('recall@top', 0):.6f}")
        print(f"   Precision@Top: {last_val.get('precision@top', 0):.6f}")
    
    # 检查训练历史
    train_history = data.get('train_history', [])
    if train_history:
        last_train = train_history[-1]
        print(f"\n6. 最后一个训练epoch:")
        print(f"   损失: {last_train.get('loss', 0):.6f}")
        print(f"   分类损失: {last_train.get('loss_class', 0):.6f}")
        print(f"   回归损失: {last_train.get('loss_reg', 0):.6f}")
    
    print("\n" + "=" * 60)
    print("建议:")
    print("=" * 60)
    print("1. 检查测试集大小和正样本比例")
    print("2. 检查损失计算逻辑（特别是测试集评估时）")
    print("3. 检查模型预测的分布（是否所有预测都相同）")
    print("4. 检查数据加载和预处理是否正确")
    print("5. 检查指标计算函数是否正确")

if __name__ == "__main__":
    metrics_file = Path(__file__).parent.parent / "artifacts_forecast" / "forecast_metrics.json"
    if metrics_file.exists():
        analyze_metrics(metrics_file)
    else:
        print(f"指标文件不存在: {metrics_file}")
        print("请先运行训练脚本生成指标文件")

