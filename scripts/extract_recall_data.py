import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

def extract_recall_data(log_dir):
    """
    从日志文件中提取召回率数据
    """
    data = []
    special_models = []
    
    # 获取所有epoch日志文件
    log_files = glob(os.path.join(log_dir, "model_epoch_*.log"))
    special_files = [
        ("best_model.log", "Best Model"),
        ("last_model.log", "Last Model")
    ]
    
    # 处理常规epoch文件
    for log_file in sorted(log_files):
        # 提取epoch编号
        filename = os.path.basename(log_file)
        epoch_match = re.search(r'model_epoch_(\d+)\.log', filename)
        
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            recall_data = extract_single_file(log_file, epoch_num, f"Epoch {epoch_num:02d}")
            if recall_data:
                data.append(recall_data)
    
    # 处理特殊模型文件
    for special_file, model_name in special_files:
        special_file_path = os.path.join(log_dir, special_file)
        if os.path.exists(special_file_path):
            recall_data = extract_single_file(special_file_path, len(data), model_name)
            if recall_data:
                special_models.append(recall_data)
    
    return data, special_models

def extract_single_file(log_file, index, model_name):
    """
    从单个日志文件中提取召回率数据
    """
    try:
        with open(log_file, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # 尝试多种模式匹配召回率信息
            patterns = [
                r"Recalls on pitts30k/queries:\s*R@1:\s*([\d\.]+),\s*R@5:\s*([\d\.]+),\s*R@10:\s*([\d\.]+),\s*R@20:\s*([\d\.]+)",
                r"Recalls for queries:\s*R@1:\s*([\d\.]+),\s*R@5:\s*([\d\.]+),\s*R@10:\s*([\d\.]+),\s*R@20:\s*([\d\.]+)",
                r"Combined recalls:\s*R@1:\s*([\d\.]+),\s*R@5:\s*([\d\.]+),\s*R@10:\s*([\d\.]+),\s*R@20:\s*([\d\.]+)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, content)
                if match:
                    # 提取召回率值
                    r1 = float(match.group(1))
                    r5 = float(match.group(2))
                    r10 = float(match.group(3))
                    r20 = float(match.group(4))
                    
                    return {
                        'Index': index,
                        'Model': model_name,
                        'Epoch': extract_epoch_number(model_name),
                        'R@1': r1,
                        'R@5': r5,
                        'R@10': r10,
                        'R@20': r20,
                        'Type': 'Special' if 'Best' in model_name or 'Last' in model_name else 'Epoch'
                    }
            
            print(f"在文件 {os.path.basename(log_file)} 中未找到召回率信息")
            return None
            
    except Exception as e:
        print(f"读取文件 {os.path.basename(log_file)} 时出错: {e}")
        return None

def extract_epoch_number(model_name):
    """
    从模型名称中提取epoch编号
    """
    if 'Epoch' in model_name:
        match = re.search(r'Epoch\s*(\d+)', model_name)
        if match:
            return int(match.group(1))
    return -1  # 特殊模型返回-1

def create_excel_report(data, special_models,  log_dir):
    """
    创建Excel报告
    """
    # 合并所有数据
    all_data = data + special_models
    
    if not all_data:
        print("没有找到有效数据")
        return None, None
    
    df = pd.DataFrame(all_data)
    
    # 添加统计信息
    epoch_data = [d for d in all_data if d['Type'] == 'Epoch']
    if epoch_data:
        epoch_df = pd.DataFrame(epoch_data)
        stats = {
            '指标': ['最大值', '最小值', '平均值', '最终值', '最佳模型', '最后模型'],
            'R@1': [
                epoch_df['R@1'].max(),
                epoch_df['R@1'].min(),
                epoch_df['R@1'].mean(),
                epoch_df.iloc[-1]['R@1'] if len(epoch_df) > 0 else 0,
                next((d['R@1'] for d in all_data if d['Model'] == 'Best Model'), 'N/A'),
                next((d['R@1'] for d in all_data if d['Model'] == 'Last Model'), 'N/A')
            ],
            'R@5': [
                epoch_df['R@5'].max(),
                epoch_df['R@5'].min(),
                epoch_df['R@5'].mean(),
                epoch_df.iloc[-1]['R@5'] if len(epoch_df) > 0 else 0,
                next((d['R@5'] for d in all_data if d['Model'] == 'Best Model'), 'N/A'),
                next((d['R@5'] for d in all_data if d['Model'] == 'Last Model'), 'N/A')
            ],
            'R@10': [
                epoch_df['R@10'].max(),
                epoch_df['R@10'].min(),
                epoch_df['R@10'].mean(),
                epoch_df.iloc[-1]['R@10'] if len(epoch_df) > 0 else 0,
                next((d['R@10'] for d in all_data if d['Model'] == 'Best Model'), 'N/A'),
                next((d['R@10'] for d in all_data if d['Model'] == 'Last Model'), 'N/A')
            ],
            'R@20': [
                epoch_df['R@20'].max(),
                epoch_df['R@20'].min(),
                epoch_df['R@20'].mean(),
                epoch_df.iloc[-1]['R@20'] if len(epoch_df) > 0 else 0,
                next((d['R@20'] for d in all_data if d['Model'] == 'Best Model'), 'N/A'),
                next((d['R@20'] for d in all_data if d['Model'] == 'Last Model'), 'N/A')
            ]
        }
    else:
        stats = {'指标': ['无epoch数据']}
    
    stats_df = pd.DataFrame(stats)
    # 确保输出目录存在
    os.makedirs(log_dir, exist_ok=True)
    output_file = os.path.join(log_dir, "model_recalls_summary.xlsx")
    # 使用ExcelWriter创建多sheet的Excel文件
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='所有模型数据', index=False)
        stats_df.to_excel(writer, sheet_name='统计信息', index=False)
        
        # 单独保存epoch数据
        if epoch_data:
            epoch_df = pd.DataFrame(epoch_data)
            epoch_df.to_excel(writer, sheet_name='Epoch数据', index=False)
    
    return df, stats_df

def plot_recall_curves(data, special_models, log_dir):
    """
    绘制召回率曲线图
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # 设置颜色和标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    metrics = ['R@1', 'R@5', 'R@10', 'R@20']
    
    # 提取epoch数据（用于绘制连续曲线）
    epoch_data = [d for d in data if d['Type'] == 'Epoch']
    epoch_data.sort(key=lambda x: x['Epoch'])
    
    if epoch_data:
        epochs = [d['Epoch'] for d in epoch_data]
        
        # 主图 - 所有指标在一起
        for i, metric in enumerate(metrics):
            values = [d[metric] for d in epoch_data]
            ax1.plot(epochs, values, 
                    marker=markers[i], linewidth=2.5, 
                    label=metric, color=colors[i], 
                    markersize=6, alpha=0.8)
        
        # 标记特殊模型点
        special_colors = {'Best Model': 'red', 'Last Model': 'blue'}
        for special_model in special_models:
            model_name = special_model['Model']
            color = special_colors.get(model_name, 'green')
            marker = 'X' if model_name == 'Best Model' else '*'
            size = 100 if model_name == 'Best Model' else 80
            
            for i, metric in enumerate(metrics):
                ax1.scatter(special_model['Epoch'] if special_model['Epoch'] != -1 else len(epoch_data) - 1, 
                           special_model[metric], 
                           color=color, marker=marker, s=size, 
                           label=f'{model_name} {metric}' if i == 0 else "", 
                           edgecolors='black', linewidth=1, zorder=5)
    
    ax1.set_title('模型在pitts30k数据集上的召回率曲线（包含最佳和最后模型）', fontsize=16, pad=20)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('召回率 (%)', fontsize=12)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    if epoch_data:
        ax1.set_xticks(epochs[::2])  # 每隔2个epoch显示一个刻度
        # 设置y轴范围
        all_values = []
        for metric in metrics:
            all_values.extend([d[metric] for d in epoch_data])
            all_values.extend([sm[metric] for sm in special_models])
        y_min = min(all_values) - 1
        y_max = max(all_values) + 1
        ax1.set_ylim(y_min, y_max)
    
    # 子图 - 特殊模型对比
    if special_models:
        models = [sm['Model'] for sm in special_models]
        x_pos = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [sm[metric] for sm in special_models]
            ax2.bar(x_pos + i*width - width*1.5, values, width, 
                   label=metric, color=colors[i], alpha=0.8)
        
        ax2.set_title('最佳模型和最后模型性能对比', fontsize=14, pad=15)
        ax2.set_xlabel('模型类型', fontsize=12)
        ax2.set_ylabel('召回率 (%)', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(models)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    # 确保输出目录存在
    os.makedirs(log_dir, exist_ok=True)
    output_image = os.path.join(log_dir, "model_recalls_curve.png")
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """
    主函数
    """
    log_dir = "./eval_results/baseline_03_lora01/sped"
    
    # 检查目录是否存在
    if not os.path.exists(log_dir):
        print(f"错误: 目录 '{log_dir}' 不存在")
        return
    
    print("开始处理日志文件...")
    
    # 提取数据
    data, special_models = extract_recall_data(log_dir)
    
    if not data and not special_models:
        print("未找到任何召回率数据")
        return
    
    print(f"成功处理 {len(data)} 个epoch的日志文件")
    print(f"成功处理 {len(special_models)} 个特殊模型文件")
    
    # 创建Excel报告
    df, stats_df = create_excel_report(data, special_models, log_dir)
    
    print(f"\nExcel报告已生成: {os.path.join(log_dir, 'model_recalls_summary.xlsx')}")
    
    # 显示数据预览
    print("\n所有模型数据预览:")
    print(df[['Model', 'R@1', 'R@5', 'R@10', 'R@20']].head(15))
    
    # 绘制图表
    print("\n生成图表...")
    plot_recall_curves(data, special_models,log_dir)
    print(f"图表已生成: {os.path.join(log_dir, 'model_recalls_curve.png')}")
    
    # 显示详细性能对比
    print(f"\n=== 性能对比分析 ===")
    
    if data:
        best_r1_epoch = max(data, key=lambda x: x['R@1']) if data else None
        best_r5_epoch = max(data, key=lambda x: x['R@5']) if data else None
        best_r10_epoch = max(data, key=lambda x: x['R@10']) if data else None
        best_r20_epoch = max(data, key=lambda x: x['R@20']) if data else None
        
        print(f"\nEpoch模型最佳性能:")
        if best_r1_epoch:
            print(f"最佳R@1: {best_r1_epoch['R@1']:.3f}% ({best_r1_epoch['Model']})")
        if best_r5_epoch:
            print(f"最佳R@5: {best_r5_epoch['R@5']:.3f}% ({best_r5_epoch['Model']})")
        if best_r10_epoch:
            print(f"最佳R@10: {best_r10_epoch['R@10']:.3f}% ({best_r10_epoch['Model']})")
        if best_r20_epoch:
            print(f"最佳R@20: {best_r20_epoch['R@5']:.3f}% ({best_r20_epoch['Model']})")
    
    if special_models:
        print(f"\n特殊模型性能:")
        for model in special_models:
            print(f"{model['Model']}: R@1: {model['R@1']:.3f}%, R@5: {model['R@5']:.3f}%, R@10: {model['R@10']:.3f}%, R@20: {model['R@20']:.3f}%")

if __name__ == "__main__":
    main()