#2026年3月18日 18点57分
#!/usr/bin/env python3
import json
import argparse
import numpy as np
import os


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
    """
    获取最佳正样本的排名。
    兼容以下情况：
    - best_positive_rank = int
    - best_positive_rank = [rank, index, dist]
    - best_positive_rank = None
    """
    rank_val = item.get('best_positive_rank')

    if rank_val is None:
        return 10000

    if isinstance(rank_val, (list, tuple)):
        if len(rank_val) > 0 and rank_val[0] is not None:
            return int(rank_val[0])
        return 10000

    return int(rank_val)


def get_dist(item):
    """安全读取 pred_geo_dist"""
    dist = item.get('pred_geo_dist')
    if dist is None:
        return None
    try:
        return float(dist)
    except Exception:
        return None


def summarize_distance_changes(keys, base_map, imp_map):
    """
    对一组样本做距离变化统计
    delta = base_dist - imp_dist
    delta > 0 代表改进后更近
    """
    deltas = []
    improved = 0
    worsened = 0
    unchanged = 0

    base_dists = []
    imp_dists = []

    for k in keys:
        b_dist = get_dist(base_map[k])
        i_dist = get_dist(imp_map[k])

        if b_dist is None or i_dist is None:
            continue

        delta = b_dist - i_dist
        deltas.append(delta)
        base_dists.append(b_dist)
        imp_dists.append(i_dist)

        if delta > 5.0:
            improved += 1
        elif delta < -5.0:
            worsened += 1
        else:
            unchanged += 1

    if len(deltas) == 0:
        return None

    deltas = np.array(deltas, dtype=np.float32)
    base_dists = np.array(base_dists, dtype=np.float32)
    imp_dists = np.array(imp_dists, dtype=np.float32)

    summary = {
        'count_valid': len(deltas),
        'base_mean_dist': float(np.mean(base_dists)),
        'imp_mean_dist': float(np.mean(imp_dists)),
        'base_median_dist': float(np.median(base_dists)),
        'imp_median_dist': float(np.median(imp_dists)),
        'mean_delta': float(np.mean(deltas)),         # 正值越大越好
        'median_delta': float(np.median(deltas)),
        'max_improve': float(np.max(deltas)),
        'max_worsen': float(np.min(deltas)),
        'improved_count': improved,
        'worsened_count': worsened,
        'unchanged_count': unchanged
    }
    return summary


def analyze(baseline_path, improved_path, output_dir):
    base_map, base_recalls = load_results(baseline_path)
    imp_map, imp_recalls = load_results(improved_path)

    os.makedirs(output_dir, exist_ok=True)

    # 统计容器
    stats = {
        'total': 0,
        'fixed': [],          # Base错 -> Imp对 (Top1)
        'new_wrong': [],      # Base对 -> Imp错 (Top1)
        'still_wrong': [],    # 都错
        'both_correct': [],   # 都对
        'rank_improved': [],  # Rank 变好了
        'dist_improved': [],  # 距离改善 > 5m
        'dist_worsened': []   # 距离变差 > 5m
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

        # Rank 变化
        b_rank = get_rank(b_item)
        i_rank = get_rank(i_item)
        if i_rank < b_rank:
            stats['rank_improved'].append(key)

        # 距离变化
        b_dist = get_dist(b_item)
        i_dist = get_dist(i_item)
        if b_dist is not None and i_dist is not None:
            if i_dist < b_dist - 5.0:
                stats['dist_improved'].append(key)
            elif i_dist > b_dist + 5.0:
                stats['dist_worsened'].append(key)

    # ---------- 生成报告 ----------
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("COMPARISON REPORT")
    report_lines.append(f"Baseline: {os.path.basename(baseline_path)}")
    report_lines.append(f"Improved: {os.path.basename(improved_path)}")
    report_lines.append("=" * 60)

    # 1. Recall 对比
    report_lines.append("\n[Global Recall Metrics]")
    report_lines.append(f"{'Metric':<10} {'Baseline':<10} {'Improved':<10} {'Delta':<10}")
    report_lines.append("Recalls (Raw Strings):")
    for k in base_recalls.keys():
        report_lines.append(f"  Query Set: {k}")
        report_lines.append(f"    Base: {base_recalls[k]}")
        report_lines.append(f"    Imp : {imp_recalls.get(k, 'N/A')}")

    # 2. Top-1 分类统计
    report_lines.append("\n[Classification Analysis (Top-1)]")
    report_lines.append(f"Total Queries: {stats['total']}")
    report_lines.append(f"✅ Fixed (Base X -> Imp O):       {len(stats['fixed'])} ({len(stats['fixed'])/stats['total']:.2%})")
    report_lines.append(f"❌ New Wrong (Base O -> Imp X):   {len(stats['new_wrong'])} ({len(stats['new_wrong'])/stats['total']:.2%})")
    report_lines.append(f"💀 Still Wrong (Base X -> Imp X): {len(stats['still_wrong'])} ({len(stats['still_wrong'])/stats['total']:.2%})")
    report_lines.append(f"🎉 Both Correct:                  {len(stats['both_correct'])} ({len(stats['both_correct'])/stats['total']:.2%})")

    net_gain = len(stats['fixed']) - len(stats['new_wrong'])
    report_lines.append(f"👉 Net Improvement (Fixed - Broken): {net_gain}")

    # 3. Rank 深度分析
    report_lines.append("\n[Deep Dive: Rank Analysis]")

    rank_gains = []
    for k in stats['fixed']:
        b_rank = get_rank(base_map[k])
        i_rank = get_rank(imp_map[k])
        rank_gains.append(b_rank - i_rank)

    if rank_gains:
        avg_rank_gain = np.mean(rank_gains)
        med_rank_gain = np.median(rank_gains)
        report_lines.append(f"For 'Fixed' samples, average rank improved by: {avg_rank_gain:.2f} positions")
        report_lines.append(f"For 'Fixed' samples, median rank improved by:  {med_rank_gain:.2f} positions")

    still_wrong_rank_imp = 0
    for k in stats['still_wrong']:
        if k in stats['rank_improved']:
            still_wrong_rank_imp += 1

    report_lines.append(f"Inside 'Still Wrong' ({len(stats['still_wrong'])} samples):")
    report_lines.append(f"  - {still_wrong_rank_imp} samples have better GT ranking than baseline")

    # 4. Distance 深度分析
    report_lines.append("\n[Deep Dive: Distance Analysis]")

    global_dist_summary = summarize_distance_changes(common_keys, base_map, imp_map)
    if global_dist_summary is not None:
        report_lines.append("[Overall Distance Change]")
        report_lines.append(f"  Valid samples with distance info: {global_dist_summary['count_valid']}")
        report_lines.append(f"  Baseline mean dist   : {global_dist_summary['base_mean_dist']:.2f} m")
        report_lines.append(f"  Improved mean dist   : {global_dist_summary['imp_mean_dist']:.2f} m")
        report_lines.append(f"  Baseline median dist : {global_dist_summary['base_median_dist']:.2f} m")
        report_lines.append(f"  Improved median dist : {global_dist_summary['imp_median_dist']:.2f} m")
        report_lines.append(f"  Mean delta (Base-Imp): {global_dist_summary['mean_delta']:.2f} m")
        report_lines.append(f"  Median delta         : {global_dist_summary['median_delta']:.2f} m")
        report_lines.append(f"  Improved (>5m)       : {global_dist_summary['improved_count']}")
        report_lines.append(f"  Worsened  (>5m)      : {global_dist_summary['worsened_count']}")
        report_lines.append(f"  Nearly unchanged     : {global_dist_summary['unchanged_count']}")
        report_lines.append(f"  Best improvement     : {global_dist_summary['max_improve']:.2f} m")
        report_lines.append(f"  Worst regression     : {global_dist_summary['max_worsen']:.2f} m")

    # 分类别分析
    category_groups = {
        'Fixed': stats['fixed'],
        'New Wrong': stats['new_wrong'],
        'Still Wrong': stats['still_wrong'],
        'Both Correct': stats['both_correct']
    }

    report_lines.append("\n[Distance Change by Category]")
    for cat_name, keys in category_groups.items():
        summary = summarize_distance_changes(keys, base_map, imp_map)
        if summary is None:
            report_lines.append(f"  {cat_name}: No valid distance data")
            continue

        report_lines.append(f"  {cat_name} ({summary['count_valid']} valid samples):")
        report_lines.append(f"    Base mean dist   : {summary['base_mean_dist']:.2f} m")
        report_lines.append(f"    Imp mean dist    : {summary['imp_mean_dist']:.2f} m")
        report_lines.append(f"    Mean delta       : {summary['mean_delta']:.2f} m")
        report_lines.append(f"    Median delta     : {summary['median_delta']:.2f} m")
        report_lines.append(f"    Improved (>5m)   : {summary['improved_count']}")
        report_lines.append(f"    Worsened  (>5m)  : {summary['worsened_count']}")
        report_lines.append(f"    Unchanged        : {summary['unchanged_count']}")

    # 5. 对 still_wrong 中距离改善的样本单独强调
    still_wrong_dist_imp = 0
    for k in stats['still_wrong']:
        if k in stats['dist_improved']:
            still_wrong_dist_imp += 1

    report_lines.append("\n[Interpretation Hints]")
    report_lines.append(f"Inside 'Still Wrong' ({len(stats['still_wrong'])} samples):")
    report_lines.append(f"  - {still_wrong_dist_imp} samples retrieved geographically closer images")
    report_lines.append("  - This indicates the method may improve coarse localization even when Top-1 is still incorrect")

    # 6. Action items
    report_lines.append("\n[Action Items]")
    report_lines.append("1. Check 'new_wrong_details.json' to analyze regressions.")
    report_lines.append("2. Check 'dist_improved_details.json' to inspect distance improvements.")
    report_lines.append("3. Prioritize samples that are still wrong but have both better rank and better geo-distance.")

    # 写入报告
    report_path = os.path.join(output_dir, 'detailed_analysis.txt')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))
    print(f"\nReport saved to {report_path}")

    # 保存 new wrong 详情
    new_wrong_details = []
    for k in stats['new_wrong']:
        item = imp_map[k]
        new_wrong_details.append({
            'query_folder': item['query_folder'],
            'query_idx': item['query_idx'],
            'query_image': item.get('query_image_path'),
            'base_pred': base_map[k].get('pred_image_path'),
            'imp_pred': item.get('pred_image_path'),
            'gt_image': item.get('gt_image_path'),
            'base_rank': get_rank(base_map[k]),
            'imp_rank': get_rank(item),
            'base_dist': get_dist(base_map[k]),
            'imp_dist': get_dist(item),
            'dist_delta': None if (get_dist(base_map[k]) is None or get_dist(item) is None)
                         else get_dist(base_map[k]) - get_dist(item)
        })

    with open(os.path.join(output_dir, 'new_wrong_details.json'), 'w') as f:
        json.dump(new_wrong_details, f, indent=2)

    # 保存 dist_improved 详情
    dist_improved_details = []
    for k in stats['dist_improved']:
        b_item = base_map[k]
        i_item = imp_map[k]

        dist_improved_details.append({
            'query_folder': i_item['query_folder'],
            'query_idx': i_item['query_idx'],
            'query_image': i_item.get('query_image_path'),
            'gt_image': i_item.get('gt_image_path'),
            'base_pred': b_item.get('pred_image_path'),
            'imp_pred': i_item.get('pred_image_path'),
            'base_hit1': b_item.get('hit@1'),
            'imp_hit1': i_item.get('hit@1'),
            'base_rank': get_rank(b_item),
            'imp_rank': get_rank(i_item),
            'base_dist': get_dist(b_item),
            'imp_dist': get_dist(i_item),
            'dist_delta': get_dist(b_item) - get_dist(i_item)  # 正值说明距离缩短
        })

    with open(os.path.join(output_dir, 'dist_improved_details.json'), 'w') as f:
        json.dump(dist_improved_details, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True, help='Path to baseline json')
    parser.add_argument('--improved', required=True, help='Path to improved json')
    parser.add_argument('--out_dir', default='./analysis_output', help='Output directory')
    args = parser.parse_args()

    analyze(args.baseline, args.improved, args.out_dir)