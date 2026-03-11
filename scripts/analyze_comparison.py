#2026年3月9日 15点15分 分析对比实验结果
#!/usr/bin/env python3
import json
import argparse
import numpy as np
import os
from collections import defaultdict

def load_results(json_path):
    print(f"Loading: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    # 将列表转换为以 (folder, idx) 为 key 的字典，方便快速查找
    results_map = {}
    for item in data['per_query_results']:
        key = (item['query_folder'], item['query_idx'])
        results_map[key] = item
    return results_map, data['recalls']

def get_rank(item):
    """获取最佳正样本的排名，如果没有命中则返回一个大数值(例如 > top_k)"""
    # 脚本中保存了 best_positive_rank，是一个 tuple [rank, index, dist] 或 null
    # 注意：JSON中可能是列表形式保存 tuple
    rank_val = item.get('best_positive_rank')
    if rank_val is not None:
        return int(rank_val)
    return 10000 # 如果没找到，给一个很大的惩罚值

def analyze(baseline_path, improved_path, output_dir):
    base_map, base_recalls = load_results(baseline_path)
    imp_map, imp_recalls = load_results(improved_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计容器
    stats = {
        'total': 0,
        'fixed': [],       # Base错 -> Imp对 (Top1)
        'new_wrong': [],   # Base对 -> Imp错 (Top1)
        'still_wrong': [], # 都错
        'both_correct': [],# 都对
        'rank_improved': [], # Rank 变好了 (即使 R@1 没变)
        'dist_improved': []  # 预测结果地理距离更近了
    }
    
    # 遍历所有 Query
    common_keys = set(base_map.keys()) & set(imp_map.keys())
    stats['total'] = len(common_keys)
    
    for key in common_keys:
        b_item = base_map[key]
        i_item = imp_map[key]
        
        b_hit1 = b_item['hit@1']
        i_hit1 = i_item['hit@1']
        
        # 基础分类
        if not b_hit1 and i_hit1:
            stats['fixed'].append(key)
        elif b_hit1 and not i_hit1:
            stats['new_wrong'].append(key)
        elif not b_hit1 and not i_hit1:
            stats['still_wrong'].append(key)
        else:
            stats['both_correct'].append(key)
            
        # 深度分析：Rank 变化 (针对 R@1 没命中的情况看是否有改善)
        b_rank = get_rank(b_item)
        i_rank = get_rank(i_item)
        if i_rank < b_rank:
            stats['rank_improved'].append(key)
            
        # 深度分析：地理距离变化 (针对预测错误的样本，看预测点是否离真值更近了)
        # 注意：如果预测结果为空，距离可能为 None
        b_dist = b_item.get('pred_geo_dist')
        i_dist = i_item.get('pred_geo_dist')
        
        if b_dist is not None and i_dist is not None:
            # 如果改进后的预测误差比 Baseline 小至少 5米
            if i_dist < b_dist - 5.0: 
                stats['dist_improved'].append(key)

    # --- 生成报告 ---
    report_lines = []
    report_lines.append("="*50)
    report_lines.append(f"COMPARISON REPORT")
    report_lines.append(f"Baseline: {os.path.basename(baseline_path)}")
    report_lines.append(f"Improved: {os.path.basename(improved_path)}")
    report_lines.append("="*50)
    
    # 1. Recall 对比
    report_lines.append(f"\n[Global Recall Metrics]")
    report_lines.append(f"{'Metric':<10} {'Baseline':<10} {'Improved':<10} {'Delta':<10}")
    # 解析 recall 字符串 "R@1: 85.123, R@5: ..."
    # 这里简单假设用户脚本里是 dict 或者 string，这里直接打印 raw string 对比
    # 如果是 dict 最好，这里假设是之前脚本输出的 raw string
    report_lines.append(f"Recalls (Raw Strings):")
    for k in base_recalls.keys():
        report_lines.append(f"  Query Set: {k}")
        report_lines.append(f"    Base: {base_recalls[k]}")
        report_lines.append(f"    Imp : {imp_recalls.get(k, 'N/A')}")

    # 2. 核心分类统计
    report_lines.append(f"\n[Classification Analysis (Top-1)]")
    report_lines.append(f"Total Queries: {stats['total']}")
    report_lines.append(f"✅ Fixed (Base X -> Imp O):      {len(stats['fixed'])} ({len(stats['fixed'])/stats['total']:.2%})")
    report_lines.append(f"❌ New Wrong (Base O -> Imp X):  {len(stats['new_wrong'])} ({len(stats['new_wrong'])/stats['total']:.2%})")
    report_lines.append(f"💀 Still Wrong (Base X -> Imp X):{len(stats['still_wrong'])} ({len(stats['still_wrong'])/stats['total']:.2%})")
    report_lines.append(f"🎉 Both Correct:                 {len(stats['both_correct'])}")
    
    net_gain = len(stats['fixed']) - len(stats['new_wrong'])
    report_lines.append(f"👉 Net Improvement (Fixed - Broken): {net_gain}")

    # 3. 深度指标
    report_lines.append(f"\n[Deep Dive Analysis]")
    
    # 分析 Fixed 样本的提升幅度
    rank_gains = []
    for k in stats['fixed']:
        b_rank = get_rank(base_map[k])
        i_rank = get_rank(imp_map[k])
        rank_gains.append(b_rank - i_rank)
    if rank_gains:
        avg_rank_gain = np.mean(rank_gains)
        report_lines.append(f"For 'Fixed' samples, average rank improved by: {avg_rank_gain:.1f} positions")

    # 分析 Still Wrong 样本
    still_wrong_rank_imp = 0
    still_wrong_dist_imp = 0
    for k in stats['still_wrong']:
        if k in stats['rank_improved']: still_wrong_rank_imp += 1
        if k in stats['dist_improved']: still_wrong_dist_imp += 1
    
    report_lines.append(f"Inside 'Still Wrong' ({len(stats['still_wrong'])} samples):")
    report_lines.append(f"  - {still_wrong_rank_imp} samples have better ranking GT than baseline (Potential win)")
    report_lines.append(f"  - {still_wrong_dist_imp} samples retrieved geographically closer images (Visual improvement)")

    # 4. 输出 New Wrong 列表供 Debug
    report_lines.append(f"\n[Action Items]")
    report_lines.append(f"Check 'new_wrong_details.json' to analyze regressions.")

    # 写入文件
    report_path = os.path.join(output_dir, 'detailed_analysis.txt')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    
    print("\n".join(report_lines))
    print(f"\nReport saved to {report_path}")

    # 保存 New Wrong 的详细信息以便查看图片
    new_wrong_details = []
    for k in stats['new_wrong']:
        item = imp_map[k]
        new_wrong_details.append({
            'query_idx': item['query_idx'],
            'query_image': item['query_image_path'],
            'base_pred': base_map[k]['pred_image_path'],
            'imp_pred': item['pred_image_path'],
            'gt_image': item['gt_image_path'],
            'base_rank': get_rank(base_map[k]),
            'imp_rank': get_rank(item)
        })
    
    with open(os.path.join(output_dir, 'new_wrong_details.json'), 'w') as f:
        json.dump(new_wrong_details, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True, help='Path to baseline json')
    parser.add_argument('--improved', required=True, help='Path to improved json')
    parser.add_argument('--out_dir', default='./analysis_output', help='Output directory')
    args = parser.parse_args()
    
    analyze(args.baseline, args.improved, args.out_dir)