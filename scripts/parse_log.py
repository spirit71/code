import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np

def parse_and_plot_v2(file_path):
    # 正则表达式定义
    # 1. 召回率 (保持之前的逻辑)
    # Recalls on val set ... R@1: 95.386 ...
    recall_pattern = re.compile(r"Recalls on val set.*R@1:\s*([\d\.]+).*R@5:\s*([\d\.]+)")
    
    # 2. Alpha Mean (根据您提供的示例)
    # > Alpha Mean (整体微调强度): 0.212039
    alpha_mean_pattern = re.compile(r"Alpha Mean\s*\(整体微调强度\):\s*([\d\.]+)")
    
    # 3. 最活跃层 (根据您提供的示例)
    # > 最活跃层: Index 7 (强度: 0.2706)
    layer_pattern = re.compile(r"最活跃层:\s*Index\s*(\d+)\s*\(强度:\s*([\d\.]+)\)")
    
    # 4. Epoch 锚点
    # Start training epoch: 51
    # Model weights for epoch 50 saved
    epoch_start_pattern = re.compile(r"Start training epoch:\s*(\d+)")
    epoch_save_pattern = re.compile(r"Model weights for epoch\s*(\d+)\s*saved")

    data_list = []
    
    # 状态变量
    current_epoch = 0 # 追踪正在训练的 Epoch
    temp_r1 = None
    temp_r5 = None
    temp_alpha_mean = None
    temp_layer_idx = None
    temp_layer_val = None
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        # --- Epoch 追踪 ---
        # 遇到 "Start training epoch: X"，更新当前 epoch 计数器
        start_match = epoch_start_pattern.search(line)
        if start_match:
            # 在进入新 Epoch 前，如果之前的数据未保存（例如 Crash 了），这里可以做处理
            # 但通常我们以 "Model saved" 为保存点
            current_epoch = int(start_match.group(1))
            
            # 清空临时变量，准备接收新一轮训练过程中的 Alpha 数据
            temp_alpha_mean = None
            temp_layer_idx = None
            temp_layer_val = None
            continue

        # --- Alpha 数据提取 ---
        # 注意：这些数据通常在 Epoch 训练过程中打印
        alpha_match = alpha_mean_pattern.search(line)
        if alpha_match:
            temp_alpha_mean = float(alpha_match.group(1))
            
        layer_match = layer_pattern.search(line)
        if layer_match:
            temp_layer_idx = int(layer_match.group(1))
            temp_layer_val = float(layer_match.group(2))

        # --- Recall 数据提取 ---
        # 验证集通常在 Epoch 结束时运行
        recall_match = recall_pattern.search(line)
        if recall_match:
            temp_r1 = float(recall_match.group(1))
            temp_r5 = float(recall_match.group(2))

        # --- 数据归档 ---
        # 遇到 "Model weights for epoch X saved"，说明该 Epoch 完成
        save_match = epoch_save_pattern.search(line)
        if save_match:
            saved_epoch = int(save_match.group(1))
            
            # 将收集到的数据打包
            # 注意：如果 Alpha 数据是在 validation 之后打印的，可能需要调整逻辑
            # 这里假设 Alpha 在训练中打印，Recall 在最后打印
            
            # 如果某些值缺失（比如日志截断），填入 NaN 或跳过
            entry = {
                'epoch': saved_epoch,
                'R@1': temp_r1 if temp_r1 else np.nan,
                'Alpha Mean': temp_alpha_mean if temp_alpha_mean is not None else np.nan,
                'Max Layer Index': temp_layer_idx if temp_layer_idx is not None else np.nan,
                'Max Layer Value': temp_layer_val if temp_layer_val is not None else np.nan
            }
            data_list.append(entry)
            
            # 重置 Recall，Alpha 数据在 Start epoch 处重置
            temp_r1 = None
            temp_r5 = None

    df = pd.DataFrame(data_list)
    
    # 简单的后处理：如果某些 Epoch 只有 Recall 没有 Alpha（可能是日志格式不统一），
    # 或者反之，我们使用 interpolate 或仅绘制存在的点。
    # 这里直接绘图，matplotlib 会自动处理 NaN。

    if df.empty:
        return "未提取到有效数据，请检查日志内容是否包含指定格式的中文字符串。", None

    # --- 绘图 ---
    plt.style.use('bmh') # 使用一种整洁的样式
    
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1, hspace=0.3)

    # 子图 1: Recall R@1
    ax1 = fig.add_subplot(gs[0])
    # 过滤掉 R@1 为空的行进行绘制
    df_r1 = df.dropna(subset=['R@1'])
    ax1.plot(df_r1['epoch'], df_r1['R@1'], color='#1f77b4', marker='o', linewidth=2, label='R@1 Recall')
    
    # 标注最大值
    if not df_r1.empty:
        max_r1 = df_r1['R@1'].max()
        max_r1_ep = df_r1.loc[df_r1['R@1'].idxmax(), 'epoch']
        ax1.annotate(f'Max: {max_r1:.2f}% (Ep {int(max_r1_ep)})',
                     xy=(max_r1_ep, max_r1), xytext=(max_r1_ep, max_r1 - 0.5),
                     arrowprops=dict(facecolor='black', shrink=0.05))
    
    ax1.set_ylabel('Recall R@1 (%)', fontsize=12)
    ax1.set_title('Validation Recall (R@1)', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='lower right')

    # 子图 2: Alpha Mean & Max Layer Value (强度对比)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    df_alpha = df.dropna(subset=['Alpha Mean'])
    
    # 绘制 Alpha Mean (整体)
    ax2.plot(df_alpha['epoch'], df_alpha['Alpha Mean'], color='#d62728', marker='d', linestyle='-', label='Alpha Mean (Overall)')
    
    # 绘制 Max Layer Value (局部最强)
    ax2.plot(df_alpha['epoch'], df_alpha['Max Layer Value'], color='#ff7f0e', marker='.', linestyle='--', alpha=0.8, label='Max Layer Intensity')
    
    ax2.set_ylabel('Intensity Value', fontsize=12)
    ax2.set_title('Fine-tuning Intensity: Overall Mean vs. Most Active Layer', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 子图 3: 最活跃层索引 (Layer Index)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # 使用散点图，因为层索引是离散的
    ax3.scatter(df_alpha['epoch'], df_alpha['Max Layer Index'], color='#9467bd', s=50, label='Most Active Layer Index', zorder=5)
    ax3.step(df_alpha['epoch'], df_alpha['Max Layer Index'], where='mid', color='#9467bd', alpha=0.3, zorder=1) # 辅助连线
    
    ax3.set_ylabel('Layer Index (0-11)', fontsize=12)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_title('Most Active Layer Index per Epoch', fontsize=14, fontweight='bold')
    ax3.set_yticks(range(0, 12)) # 假设是 ViT Base 12层
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('analysis_chart.png')
    
    return "解析成功，图表已生成。", df

msg, df_result = parse_and_plot_v2('./logs/default/2025-12-17_03-28-23/debug.log')
print(msg)
if df_result is not None:
    # 打印最后几行数据供用户检查
    print(df_result[['epoch', 'R@1', 'Alpha Mean', 'Max Layer Index']].tail())