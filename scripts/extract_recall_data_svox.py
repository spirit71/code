
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

# 设置中文字体 (防止中文乱码)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK JP', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def extract_recall_data(log_dir):
    """
    遍历日志目录提取数据
    """
    data = []
    special_models = []
    
    # 1. 获取所有 model_epoch_*.log
    log_files = glob(os.path.join(log_dir, "model_epoch_*.log"))
    
    # 2. 定义特殊模型文件
    special_files = [
        ("best_model.log", "Best Model"),
        ("last_model.log", "Last Model")
    ]
    
    # 3. 处理常规 Epoch 文件
    for log_file in sorted(log_files):
        filename = os.path.basename(log_file)
        epoch_match = re.search(r'model_epoch_(\d+)\.log', filename)
        
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            recall_data = extract_single_file(log_file, epoch_num, f"Epoch {epoch_num:02d}")
            if recall_data:
                data.append(recall_data)
    
    # 4. 处理特殊模型文件 (Best/Last)
    for special_file, model_name in special_files:
        special_file_path = os.path.join(log_dir, special_file)
        if os.path.exists(special_file_path):
            recall_data = extract_single_file(special_file_path, len(data), model_name)
            if recall_data:
                special_models.append(recall_data)
    
    return data, special_models

def extract_single_file(log_file, index, model_name):
    """
    从单个日志文件中解析 SVOX 格式的召回率
    """
    try:
        with open(log_file, 'r', encoding='utf-8') as file:
            content = file.read()
            
        result = {
            'Index': index,
            'Model': model_name,
            'Epoch': extract_epoch_number(model_name),
            'Type': 'Special' if 'Best' in model_name or 'Last' in model_name else 'Epoch'
        }
        
        found_any = False

        # --- 1. 匹配各个子集 (queries, queries_night, queries_rain 等) ---
        # 模式匹配: "Recalls for <subset_name>: R@1: ..., R@5: ..."
        subset_pattern = re.compile(
            r"Recalls for ([\w_]+):\s*R@1:\s*([\d\.]+),\s*R@5:\s*([\d\.]+),\s*R@10:\s*([\d\.]+),\s*R@20:\s*([\d\.]+)"
        )
        
        subset_matches = subset_pattern.findall(content)
        for match in subset_matches:
            subset_name = match[0]  # e.g., queries, queries_night
            result[f"{subset_name}/R@1"] = float(match[1])
            result[f"{subset_name}/R@5"] = float(match[2])
            result[f"{subset_name}/R@10"] = float(match[3])
            result[f"{subset_name}/R@20"] = float(match[4])
            found_any = True

        # --- 2. 匹配 Combined recalls ---
        # 模式匹配: "Combined recalls: R@1: ..., R@5: ..."
        combined_pattern = re.compile(
            r"Combined recalls:\s*R@1:\s*([\d\.]+),\s*R@5:\s*([\d\.]+),\s*R@10:\s*([\d\.]+),\s*R@20:\s*([\d\.]+)"
        )
        
        combined_match = combined_pattern.search(content)
        if combined_match:
            result["Combined/R@1"] = float(combined_match.group(1))
            result["Combined/R@5"] = float(combined_match.group(2))
            result["Combined/R@10"] = float(combined_match.group(3))
            result["Combined/R@20"] = float(combined_match.group(4))
            found_any = True

        if not found_any:
            print(f"警告: 在文件 {os.path.basename(log_file)} 中未找到符合 SVOX 格式的数据")
            return None
            
        return result

    except Exception as e:
        print(f"读取文件 {os.path.basename(log_file)} 时出错: {e}")
        return None

def extract_epoch_number(model_name):
    if 'Epoch' in model_name:
        match = re.search(r'Epoch\s*(\d+)', model_name)
        if match:
            return int(match.group(1))
    return -1

def create_excel_report(data, special_models, log_dir):
    """
    生成包含所有场景数据的 Excel 报告
    """
    all_data = data + special_models
    if not all_data:
        return None

    df = pd.DataFrame(all_data)
    
    # 调整列顺序: 把基本信息放前面，Combined放中间，其他放后面
    cols = list(df.columns)
    base_cols = ['Index', 'Model', 'Epoch', 'Type']
    metric_cols = [c for c in cols if c not in base_cols]
    
    # 将 Combined 开头的列排在 metric 的最前面，其他按字母顺序排
    combined_cols = sorted([c for c in metric_cols if c.startswith('Combined')])
    other_metric_cols = sorted([c for c in metric_cols if not c.startswith('Combined')])
    
    final_cols = base_cols + combined_cols + other_metric_cols
    df = df[final_cols]

    # 保存
    os.makedirs(log_dir, exist_ok=True)
    output_file = os.path.join(log_dir, "svox_recalls_summary.xlsx")
    df.to_excel(output_file, index=False)
    print(f"Excel 报告已保存: {output_file}")
    
    return df

def plot_svox_curves(data, special_models, log_dir):
    """
    绘制 SVOX 专用图表：
    1. Combined R@1/5/10 曲线
    2. 各场景 R@1 对比 (Night vs Rain vs Sun...)
    """
    epoch_data = [d for d in data if d['Type'] == 'Epoch']
    epoch_data.sort(key=lambda x: x['Epoch'])
    
    if not epoch_data:
        print("没有 Epoch 数据用于绘图")
        return

    epochs = [d['Epoch'] for d in epoch_data]
    
    # 找出所有的 subset 名称 (去除 Combined)
    keys = epoch_data[0].keys()
    subsets = set()
    for key in keys:
        if '/' in key and not key.startswith('Combined') and not key.startswith('Index'):
            subsets.add(key.split('/')[0])
    subsets = sorted(list(subsets))  # ['queries', 'queries_night', 'queries_rain', ...]

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))

    # --- 图 1: Combined Metrics (综合性能) ---
    metrics = ['R@1', 'R@5', 'R@10']
    colors = ['#d62728', '#1f77b4', '#ff7f0e'] # 红、蓝、橙
    
    for i, metric in enumerate(metrics):
        key = f"Combined/{metric}"
        if key in epoch_data[0]:
            values = [d[key] for d in epoch_data]
            ax1.plot(epochs, values, marker='o', label=f"Combined {metric}", color=colors[i], linewidth=2)
            
            # 标注 Best Model 点
            for sm in special_models:
                if sm['Model'] == 'Best Model' and key in sm:
                    ax1.scatter(sm['Epoch'] if sm['Epoch']!=-1 else epochs[-1], sm[key], 
                                color='gold', marker='*', s=200, zorder=10, edgecolors='black')

    ax1.set_title('SVOX Combined Recalls (综合召回率) 训练曲线', fontsize=16)
    ax1.set_ylabel('Recall (%)', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- 图 2: Scenario Analysis (R@1 各场景对比) ---
    # 颜色映射，区分不同场景
    cmap = plt.get_cmap('tab10')
    
    # 首先画 Combined 作为参考基准
    if "Combined/R@1" in epoch_data[0]:
        combined_vals = [d["Combined/R@1"] for d in epoch_data]
        ax2.plot(epochs, combined_vals, color='black', linewidth=3, linestyle='--', label='Combined (Avg)', alpha=0.7)

    # 画各个子集
    for i, subset in enumerate(subsets):
        key = f"{subset}/R@1"
        if key in epoch_data[0]:
            values = [d[key] for d in epoch_data]
            # 简单的名称美化
            label_name = subset.replace('queries_', '').capitalize()
            if label_name == 'Queries': label_name = 'Day (Base)'
            
            ax2.plot(epochs, values, marker='.', label=label_name, color=cmap(i), linewidth=1.5, alpha=0.8)

    ax2.set_title('各场景 R@1 性能对比 (Scenario Analysis)', fontsize=16)
    ax2.set_ylabel('R@1 (%)', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.grid(True, alpha=0.3)
    # 将图例放在外侧，防止遮挡
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    plt.tight_layout()
    output_img = os.path.join(log_dir, "svox_performance_analysis.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {output_img}")
    plt.show()

def main():
    # log_dir = "./logs/svox_experiment"  # 请修改为实际的日志路径
    log_dir = "./eval_results/dinov2_B_experiment_01_seed0/svox"
    
    if not os.path.exists(log_dir):
        print(f"错误: 目录 {log_dir} 不存在")
        return

    print("开始提取 SVOX 日志数据...")
    data, special_models = extract_recall_data(log_dir)
    
    if not data and not special_models:
        print("未提取到有效数据，请检查日志格式。")
        return

    print(f"处理了 {len(data)} 个 Epoch 文件和 {len(special_models)} 个特殊模型文件。")

    # 生成 Excel
    df = create_excel_report(data, special_models, log_dir)

    # 打印最新一个 Epoch 的摘要
    if data:
        last_epoch = data[-1]
        print(f"\n=== Epoch {last_epoch['Epoch']} SVOX 摘要 ===")
        print(f"Combined R@1: {last_epoch.get('Combined/R@1', 'N/A')}%")
        # 打印各子集 R@1
        for k, v in last_epoch.items():
            if k.endswith('/R@1') and not k.startswith('Combined'):
                print(f"  - {k.split('/')[0]}: {v}%")

    # 绘图
    plot_svox_curves(data, special_models, log_dir)

if __name__ == "__main__":
    main()
