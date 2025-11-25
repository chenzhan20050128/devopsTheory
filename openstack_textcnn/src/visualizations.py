"""Create various visualization charts based on the summary report"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional

# Set font style for English text
import platform
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# Set chart style
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'primary': '#1976D2',
    'secondary': '#388E3C',
    'accent': '#F57C00',
    'error': '#D32F2F',
    'success': '#388E3C',
    'info': '#0288D1',
}


def draw_data_preprocessing_flow():
    """Draw data preprocessing flow diagram"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Define processing steps
    steps = [
        ("Raw Log Files", "openstack_normal1.log\nopenstack_normal2.log\nopenstack_abnormal.log", "#E3F2FD"),
        ("Anomaly Labels", "anomaly_labels.txt\n(4 anomaly instance IDs)", "#FFF3E0"),
        ("Dynamic Label Generation", "1. Anomaly ID matching\n2. Request ID propagation\n3. Keyword supplement\n4. Default normal", "#E8F5E9"),
        ("Log Text Normalization", "1. Lowercase conversion\n2. Entity replacement\n(UUID/IP/Numeric)\n3. Character cleaning", "#F3E5F5"),
        ("Numeric Feature Extraction", "Statistical features:\nCount/Mean/Max\nMin/Std", "#E0F2F1"),
        ("Sliding Window Construction", "Window size: 8\nStride: 4\nLabel aggregation", "#FFF9C4"),
        ("Final Dataset", "28,264 samples\nNormal: 83.4%\nAnomaly: 16.6%", "#FFE0B2"),
    ]
    
    # Draw flow boxes
    box_width = 2.0
    box_height = 1.2
    start_x = 1.0
    y_pos = 8.0
    
    boxes = []
    for i, (title, content, color) in enumerate(steps):
        x = start_x + i * 2.2
        y = y_pos
        
        # Draw main box
        rect = mpatches.FancyBboxPatch(
            (x - box_width/2, y - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=2,
            zorder=2
        )
        ax.add_patch(rect)
        
        # Add title
        ax.text(x, y + 0.4, title, ha='center', va='center',
                fontsize=11, fontweight='bold', zorder=3)
        
        # Add content
        ax.text(x, y - 0.2, content, ha='center', va='center',
                fontsize=9, zorder=3)
        
        boxes.append((x, y))
        
        # Draw arrow (except for the last one)
        if i < len(steps) - 1:
            arrow = mpatches.FancyArrowPatch(
                (x + box_width/2, y),
                (x + box_width/2 + 0.2, y),
                arrowstyle='->',
                mutation_scale=20,
                linewidth=2,
                color='black',
                zorder=1
            )
            ax.add_patch(arrow)
    
    # Set figure range
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_title('OpenStack Log Data Preprocessing Flow', fontsize=16, fontweight='bold', pad=20)
    
    # Save
    output_path = Path(__file__).parent / "data_preprocessing_flow.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Data preprocessing flow diagram generated: {output_path}")
    plt.close()


def draw_training_curves():
    """Draw training process curves (based on report data)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training history data (from report)
    epochs = [1, 2, 3, 4, 5]
    train_loss = [1.27, 0.95, 0.78, 0.68, 0.63]
    val_accuracy = [84.64, 86.11, 88.42, 88.63, 87.80]
    val_f1 = [0.58, 0.65, 0.68, 0.69, 0.70]
    val_auc = [0.88, 0.92, 0.94, 0.94, 0.95]
    
    # 1. Training loss curve
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_loss, 'o-', linewidth=2.5, markersize=8,
             color=COLORS['error'], label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xticks(epochs)
    
    # 2. Validation accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs, val_accuracy, 's-', linewidth=2.5, markersize=8,
             color=COLORS['success'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xticks(epochs)
    ax2.set_ylim(82, 90)
    
    # 3. F1-Score
    ax3 = axes[1, 0]
    ax3.plot(epochs, val_f1, '^-', linewidth=2.5, markersize=8,
             color=COLORS['primary'], label='Validation F1-Score')
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    ax3.set_title('Validation F1-Score', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_xticks(epochs)
    ax3.set_ylim(0.55, 0.72)
    
    # 4. AUC curve
    ax4 = axes[1, 1]
    ax4.plot(epochs, val_auc, 'd-', linewidth=2.5, markersize=8,
             color=COLORS['accent'], label='Validation AUC')
    ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('AUC', fontsize=11, fontweight='bold')
    ax4.set_title('Validation AUC', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    ax4.set_xticks(epochs)
    ax4.set_ylim(0.85, 0.96)
    
    plt.suptitle('Model Training Process Monitoring', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = Path(__file__).parent / "training_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training curves generated: {output_path}")
    plt.close()


def draw_dataset_statistics():
    """Draw dataset statistics charts"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Dataset split pie chart
    ax1 = axes[0]
    sizes = [19784, 4239, 4241]  # Train, Validation, Test
    labels = ['Train\n(19,784)', 'Validation\n(4,239)', 'Test\n(4,241)']
    colors_pie = ['#42A5F5', '#66BB6A', '#FFA726']
    explode = (0.05, 0, 0)
    
    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, labels=labels, colors=colors_pie,
        autopct='%1.1f%%', shadow=True, startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax1.set_title('Dataset Split Ratio', fontsize=13, fontweight='bold', pad=15)
    
    # 2. Class distribution bar chart
    ax2 = axes[1]
    categories = ['Train', 'Validation', 'Test']
    normal_counts = [16559, 3548, 3550]
    anomaly_counts = [3225, 691, 691]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, normal_counts, width, label='Normal',
                    color='#66BB6A', alpha=0.8)
    bars2 = ax2.bar(x + width/2, anomaly_counts, width, label='Anomaly',
                    color='#EF5350', alpha=0.8)
    
    ax2.set_xlabel('Dataset', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
    ax2.set_title('Class Distribution', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Dataset Statistics', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = Path(__file__).parent / "dataset_statistics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Dataset statistics chart generated: {output_path}")
    plt.close()


def draw_model_parameters():
    """Draw model parameter statistics chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Parameter count data (from report)
    components = ['Embedding', 'Convolution', 'Fully Connected', 'Numeric Encoder']
    param_counts = [3000000, 200000, 53000, 2000]  # Approximate
    param_counts_m = [3.0, 0.2, 0.053, 0.002]  # In millions
    
    colors_bar = ['#42A5F5', '#66BB6A', '#FFA726', '#EF5350']
    
    # 1. Bar chart (parameters in millions)
    bars = ax1.barh(components, param_counts_m, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Parameters (Millions)', fontsize=11, fontweight='bold')
    ax1.set_title('Parameter Distribution by Component', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, param_counts_m)):
        width = bar.get_width()
        ax1.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                f'{count:.3f}M\n({param_counts[i]:,})',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # 2. Pie chart (parameter percentage)
    total_params = sum(param_counts)
    percentages = [p/total_params*100 for p in param_counts]
    
    wedges, texts, autotexts = ax2.pie(
        param_counts, labels=components, colors=colors_bar,
        autopct=lambda pct: f'{pct:.1f}%',
        shadow=True, startangle=90,
        textprops={'fontsize': 10, 'fontweight': 'bold'}
    )
    ax2.set_title(f'Parameter Percentage\n(Total: {total_params/1e6:.2f}M)', 
                  fontsize=13, fontweight='bold', pad=15)
    
    plt.suptitle('TextCNN Model Parameter Statistics', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = Path(__file__).parent / "model_parameters.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Model parameter statistics chart generated: {output_path}")
    plt.close()


def draw_performance_metrics():
    """Draw model performance metrics visualization"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Test set performance metrics (from report)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    values = [88.75, 60.66, 88.13, 71.86, 95.02]  # AUC converted to percentage
    
    colors = ['#42A5F5', '#66BB6A', '#FFA726', '#EF5350', '#AB47BC']
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2, width=0.6)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Performance Metric (%)', fontsize=12, fontweight='bold')
    ax.set_title('Test Set Model Performance Metrics', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add reference lines
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, linewidth=1, label='90% Reference')
    ax.axhline(y=70, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='70% Reference')
    ax.legend(fontsize=9, loc='upper right')
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent / "performance_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Performance metrics chart generated: {output_path}")
    plt.close()


def draw_confusion_matrix_visual():
    """Draw confusion matrix visualization (based on report data)"""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Confusion matrix data (from report)
    cm = np.array([[3155, 395],   # TN, FP
                   [82, 609]])    # FN, TP
    
    # Draw heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
    ax.figure.colorbar(im, ax=ax)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]}\n({cm[i, j]/cm.sum()*100:.1f}%)',
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')
    
    # Set labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Normal', 'Predicted Anomaly'], fontsize=12, fontweight='bold')
    ax.set_yticklabels(['Actual Normal', 'Actual Anomaly'], fontsize=12, fontweight='bold')
    
    # Add axis labels
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_title('Test Set Confusion Matrix', fontsize=15, fontweight='bold', pad=15)
    
    # Add statistics
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum() * 100
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) * 100
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) * 100
    f1 = 2 * precision * recall / (precision + recall)
    
    stats_text = f'Accuracy: {accuracy:.2f}%\nPrecision: {precision:.2f}%\nRecall: {recall:.2f}%\nF1-Score: {f1:.2f}%'
    ax.text(1.5, 0.5, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent / "confusion_matrix_visual.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix visualization generated: {output_path}")
    plt.close()


def main():
    """Generate all visualization charts"""
    print("Starting to generate visualization charts...")
    
    # Ensure output directory exists
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        draw_data_preprocessing_flow()
        draw_training_curves()
        draw_dataset_statistics()
        draw_model_parameters()
        draw_performance_metrics()
        draw_confusion_matrix_visual()
        print("\nAll visualization charts generated successfully!")
    except Exception as e:
        print(f"Error generating charts: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

