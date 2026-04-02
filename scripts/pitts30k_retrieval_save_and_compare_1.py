#!/usr/bin/env python3
"""
2026年3月7日 17点19分
pitts30k 检索结果保存与改进前后对比脚本

功能：
1. 对 pitts30k 进行测试，将每个 query 的 top-k 检索结果保存到 JSON（含 query id、gt、数据库索引、
   top-k 结果、相似度分数、是否命中 GT、query/gt/pred 图像路径）
2. 自动筛选 top1 错误样本，批量保存错误样本图（query | gt | pred 并排）
3. 改进前后对比：对比两份 JSON，输出 fixed / still wrong / new wrong 并保存对比图与报告
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import faiss
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

# 项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datasets_ws
import network
import commons
from test import top_n_voting


def l2_distance_to_similarity(distance: float) -> float:
    """将 L2 距离转为相似度分数，范围约 (0, 1]，越大越相似。"""
    return 1.0 / (1.0 + float(distance))


def geo_distance_utm(utm_a: np.ndarray, utm_b: np.ndarray) -> float:
    """UTM 平面欧氏距离（与数据集 radius_neighbors 一致）。"""
    return float(np.linalg.norm(utm_a - utm_b))


def run_retrieval_and_collect_per_query(
    args,
    model,
    eval_ds,
    pca,
    top_k: int,
    dataset_root: str,
) -> Tuple[Dict[str, Tuple], List[Dict]]:
    """
    运行检索并收集每个 query 的 top-k 结果（与 test.test 逻辑一致）。
    返回 (all_recalls, per_query_results)。
    per_query_results: 每项为单条 query 的完整记录，用于保存 JSON。
    """
    test_method = args.test_method
    model.eval()
    with torch.no_grad():
        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds,
            num_workers=args.num_workers,
            batch_size=args.infer_batch_size,
            pin_memory=(args.device == "cuda"),
        )
        if test_method in ("nearest_crop", "maj_voting"):
            total_queries = sum(d["num"] for d in eval_ds.queries_data.values())
            all_features = np.empty(
                (5 * total_queries + eval_ds.database_num, args.features_dim), dtype="float32"
            )
        else:
            all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")

        for inputs, indices in tqdm(database_dataloader, ncols=100, desc="DB features"):
            features = model(inputs.to(args.device))
            features = features.cpu().numpy()
            if pca is not None:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features

        queries_features_dict = {}
        queries_indices_dict = {}
        for folder_name, data in eval_ds.queries_data.items():
            folder_paths = data["paths"]
            start_idx = eval_ds.images_paths.index(folder_paths[0])
            end_idx = start_idx + len(folder_paths)
            queries_subset_ds = Subset(eval_ds, list(range(start_idx, end_idx)))
            queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
            queries_dataloader = DataLoader(
                dataset=queries_subset_ds,
                num_workers=args.num_workers,
                batch_size=queries_infer_batch_size,
                pin_memory=(args.device == "cuda"),
            )
            if test_method in ("nearest_crop", "maj_voting"):
                folder_features = np.empty((5 * data["num"], args.features_dim), dtype="float32")
            else:
                folder_features = np.empty((data["num"], args.features_dim), dtype="float32")
            for inputs, indices in tqdm(
                queries_dataloader, ncols=100, desc=f"Query features {folder_name}"
            ):
                if test_method in ("five_crops", "nearest_crop", "maj_voting"):
                    inputs = torch.cat(tuple(inputs))
                features = model(inputs.to(args.device))
                if test_method == "five_crops":
                    features = torch.stack(torch.split(features, 5)).mean(1)
                features = features.cpu().numpy()
                if pca is not None:
                    features = pca.transform(features)
                if test_method in ("nearest_crop", "maj_voting"):
                    batch_start_idx = (indices[0].item() - start_idx) * 5
                    batch_end_idx = batch_start_idx + indices.shape[0] * 5
                    batch_indices = np.arange(batch_start_idx, batch_end_idx)
                    folder_features[batch_indices, :] = features
                else:
                    batch_indices = indices.numpy() - start_idx
                    folder_features[batch_indices, :] = features
            queries_features_dict[folder_name] = folder_features
            queries_indices_dict[folder_name] = (start_idx, end_idx)
            if test_method in ("nearest_crop", "maj_voting"):
                all_features[start_idx * 5 : end_idx * 5] = folder_features
            else:
                all_features[start_idx:end_idx] = folder_features

    database_features = all_features[: eval_ds.database_num]
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del database_features, all_features

    database_paths = eval_ds.database_paths
    database_utms = eval_ds.database_utms  # (N_db, 2) 用于地理距离
    db_num = eval_ds.database_num
    # 用于生成相对路径的 dataset test 根目录
    dataset_test_root = os.path.join(
        args.eval_datasets_folder, args.eval_dataset_name, "test"
    )
    geo_close_threshold = getattr(args, "val_positive_dist_threshold", 25)

    def make_relative_path(full_path: str) -> str:
        if full_path.startswith(dataset_test_root):
            return os.path.relpath(full_path, dataset_test_root)
        return full_path

    per_query_results: List[Dict] = []
    all_recalls = {}

    for folder_name, data in eval_ds.queries_data.items():
        queries_features = queries_features_dict[folder_name]
        start_idx, end_idx = queries_indices_dict[folder_name]
        query_paths = data["paths"]
        queries_utms = data["utms"]  # (N_queries, 2)
        positives_per_query = data["soft_positives"]
        k = max(max(args.recall_values), top_k)

        if test_method == "five_crops":
            queries_features = np.stack(
                [np.mean(queries_features[i : i + 5], axis=0) for i in range(0, len(queries_features), 5)]
            )
        if test_method in ("nearest_crop", "maj_voting"):
            queries_features_5 = queries_features.reshape(-1, 5, args.features_dim)
            distances_5, predictions_5 = [], []
            for i in range(5):
                d, p = faiss_index.search(queries_features_5[:, i, :], k)
                distances_5.append(d)
                predictions_5.append(p)
            distances_5 = np.stack(distances_5, axis=1)
            predictions_5 = np.stack(predictions_5, axis=1)
            if test_method == "maj_voting":
                distances_5 = top_n_voting("top5", predictions_5, distances_5, args.majority_weight)
                distances, predictions = faiss_index.search(queries_features_5.mean(axis=1).astype("float32"), k)
            else:
                best_crop = np.argmin(distances_5[:, :, 0], axis=1)
                predictions = predictions_5[np.arange(len(predictions_5)), best_crop]
                distances = np.zeros((predictions.shape[0], k), dtype=np.float32)
                for q in range(predictions.shape[0]):
                    distances[q, :] = distances_5[q, best_crop[q], :]
        else:
            distances, predictions = faiss_index.search(queries_features.astype("float32"), k)

        recalls = np.zeros(len(args.recall_values))
        for query_index in range(len(predictions)):
            pred = predictions[query_index]
            gt_indices = np.array(positives_per_query[query_index])
            query_utm = queries_utms[query_index]
            retrieved = pred[:top_k].tolist()
            dist_topk = distances[query_index][:top_k].tolist()
            scores = [round(l2_distance_to_similarity(d), 4) for d in dist_topk]
            hit1 = int(pred[0]) in gt_indices
            hit5 = any(int(p) in gt_indices for p in pred[:5])
            hit10 = any(int(p) in gt_indices for p in pred[: min(10, len(pred))])
            hit20 = any(int(p) in gt_indices for p in pred[: min(20, len(pred))])

            query_path = query_paths[query_index]
            pred_idx = int(pred[0]) if len(pred) > 0 else -1
            pred_path = database_paths[pred_idx] if pred_idx >= 0 else ""

            # ---------- 新增：全库排序与 query 到每个 GT 的特征相似度、最佳正样本排名 ----------
            q_feat = queries_features[query_index : query_index + 1].astype("float32")
            dist_all, indices_all = faiss_index.search(q_feat, db_num)
            dist_all = dist_all[0]  # (db_num,)
            indices_all = indices_all[0]  # (db_num,)

            gt_geo_dists_list: List[float] = []
            gt_feat_sims_list: List[float] = []
            gt_geo_nearest_idx = gt_feat_best_idx = gt_geo_farthest_idx = gt_geo_median_idx = None
            gt_geo_min = gt_geo_median = gt_geo_max = gt_geo_std = None
            gt_feat_best_sim = gt_feat_median_sim = gt_feat_worst_sim = None
            gt_feat_best_idx = None
            best_positive_rank = best_positive_db_idx = None
            best_positive_sim = best_positive_geo_dist = None
            pred_geo_dist = pred_is_geo_close = None
            gt_feat_sims_list = []  # 无 GT 时为空

            if len(gt_indices) > 0:
                # 地理：query 到每个 GT 的 UTM 距离
                gt_geo_dists_list = [
                    geo_distance_utm(query_utm, database_utms[int(g)]) for g in gt_indices
                ]
                gt_geo_dists_arr = np.array(gt_geo_dists_list)
                gt_geo_min = float(np.min(gt_geo_dists_arr))
                gt_geo_max = float(np.max(gt_geo_dists_arr))
                gt_geo_median = float(np.median(gt_geo_dists_arr))
                gt_geo_std = float(np.std(gt_geo_dists_arr)) if len(gt_geo_dists_arr) > 1 else 0.0
                # 四种 GT 索引
                gt_geo_nearest_idx = int(gt_indices[np.argmin(gt_geo_dists_arr)])
                gt_geo_farthest_idx = int(gt_indices[np.argmax(gt_geo_dists_arr)])
                sorted_pos = np.argsort(gt_geo_dists_arr)
                gt_geo_median_idx = int(gt_indices[sorted_pos[len(sorted_pos) // 2]])

                # 特征：query 到每个 GT 的 L2 距离 -> 相似度（在全库排序中查找）
                gt_feat_sims_list = []
                for g in gt_indices:
                    pos = np.where(indices_all == int(g))[0]
                    if len(pos) > 0:
                        d = float(dist_all[pos[0]])
                        gt_feat_sims_list.append(l2_distance_to_similarity(d))
                    else:
                        gt_feat_sims_list.append(0.0)
                gt_feat_sims_arr = np.array(gt_feat_sims_list)
                gt_feat_best_sim = float(np.max(gt_feat_sims_arr))
                gt_feat_worst_sim = float(np.min(gt_feat_sims_arr))
                gt_feat_median_sim = float(np.median(gt_feat_sims_arr))
                gt_feat_best_idx = int(gt_indices[np.argmax(gt_feat_sims_arr)])

                # 最佳正样本排名：在全库排序中每个 GT 的排名，取最小（最好）
                ranks = []
                for g in gt_indices:
                    pos = np.where(indices_all == int(g))[0]
                    if len(pos) > 0:
                        ranks.append((int(pos[0]) + 1, int(g), float(dist_all[pos[0]])))
                if ranks:
                    best_positive_rank, best_positive_db_idx, best_dist = min(ranks, key=lambda x: x[0])
                    best_positive_sim = round(l2_distance_to_similarity(best_dist), 4)
                    best_positive_geo_dist = geo_distance_utm(
                        query_utm, database_utms[best_positive_db_idx]
                    )

            # 预测与 query 的地理关系
            if pred_idx >= 0:
                pred_geo_dist = float(geo_distance_utm(query_utm, database_utms[pred_idx]))
                pred_is_geo_close = pred_geo_dist < geo_close_threshold

            # 四种 GT 的图像路径（便于可视化和分析）
            def path_for_gt_idx(idx):
                if idx is None:
                    return ""
                return make_relative_path(database_paths[idx])

            record = {
                "query_idx": query_index,
                "query_folder": folder_name,
                "query_path": make_relative_path(query_path),
                "gt_indices": [int(x) for x in gt_indices],
                "retrieved_indices": [int(x) for x in retrieved],
                "scores": scores,
                "hit@1": hit1,
                "hit@5": hit5,
                "hit@10": hit10,
                "hit@20": hit20,
                "query_image_path": make_relative_path(query_path),
                "pred_image_path": make_relative_path(pred_path) if pred_path else "",
                # 四种 GT 路径（图 2 五图并排用）
                "gt_geo_nearest_path": path_for_gt_idx(gt_geo_nearest_idx),
                "gt_feat_best_path": path_for_gt_idx(gt_feat_best_idx),
                "gt_geo_farthest_path": path_for_gt_idx(gt_geo_farthest_idx),
                "gt_geo_median_path": path_for_gt_idx(gt_geo_median_idx),
                # 兼容旧字段：保留“第一个 GT”为 gt_image_path，便于旧逻辑
                "gt_image_path": path_for_gt_idx(gt_geo_nearest_idx),
                # 正样本集合地理分布
                "gt_geo_dists": [round(x, 4) for x in gt_geo_dists_list],
                "gt_geo_min": round(gt_geo_min, 4) if gt_geo_min is not None else None,
                "gt_geo_median": round(gt_geo_median, 4) if gt_geo_median is not None else None,
                "gt_geo_max": round(gt_geo_max, 4) if gt_geo_max is not None else None,
                "gt_geo_std": round(gt_geo_std, 4) if gt_geo_std is not None else None,
                # 正样本集合特征分布
                "gt_feat_sims": [round(x, 4) for x in gt_feat_sims_list],
                "gt_feat_best_sim": round(gt_feat_best_sim, 4) if gt_feat_best_sim is not None else None,
                "gt_feat_median_sim": round(gt_feat_median_sim, 4) if gt_feat_median_sim is not None else None,
                "gt_feat_worst_sim": round(gt_feat_worst_sim, 4) if gt_feat_worst_sim is not None else None,
                "gt_feat_best_idx": gt_feat_best_idx,
                # 预测与 query 的地理关系
                "pred_geo_dist": round(pred_geo_dist, 4) if pred_geo_dist is not None else None,
                "pred_is_geo_close": pred_is_geo_close,
                # 正样本排序质量
                "best_positive_rank": best_positive_rank,
                "best_positive_db_idx": best_positive_db_idx,
                "best_positive_sim": best_positive_sim,
                "best_positive_geo_dist": round(best_positive_geo_dist, 4) if best_positive_geo_dist is not None else None,
            }
            per_query_results.append(record)

            for i, n in enumerate(args.recall_values):
                if np.any(np.isin(pred[:n], gt_indices)):
                    recalls[i:] += 1
                    break

        recalls = recalls / len(query_paths) * 100
        recalls_str = ", ".join(
            [f"R@{val}: {rec:.3f}" for val, rec in zip(args.recall_values, recalls)]
        )
        all_recalls[folder_name] = (recalls, recalls_str)

    return all_recalls, per_query_results


def save_error_sample_images(
    eval_ds,
    per_query_results: List[Dict],
    error_indices: List[int],
    save_dir: Path,
    dataset_test_root: str,
    max_images: int = 500,
    composite_width: int = 400,
):
    """
    将 top1 错误样本保存为 5 图并排（图 2）：
    Query | Pred | GT_geo_nearest | GT_feat_best | GT_geo_farthest
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def load_resized(path: str, size: Tuple[int, int]) -> Image.Image:
        if not path or not os.path.isfile(path):
            return Image.new("RGB", size, (128, 128, 128))
        img = Image.open(path).convert("RGB")
        img.thumbnail((composite_width, composite_width * 2), Image.Resampling.LANCZOS)
        return img

    def abs_path(rel_path: str) -> str:
        if not rel_path:
            return ""
        return os.path.join(dataset_test_root, rel_path) if not os.path.isabs(rel_path) else rel_path

    labels = ["Query", "Pred", "GT_geo_nearest", "GT_feat_best", "GT_geo_farthest"]
    keys = ["query_image_path", "pred_image_path", "gt_geo_nearest_path", "gt_feat_best_path", "gt_geo_farthest_path"]

    for i, rec_idx in enumerate(error_indices[:max_images]):
        rec = per_query_results[rec_idx]
        folder_name = rec["query_folder"]
        paths = [rec.get(k) or rec.get("query_path") if k == "query_image_path" else rec.get(k) or "" for k in keys]
        paths = [abs_path(p) for p in paths]

        images = [load_resized(p, (composite_width, composite_width)) for p in paths]
        widths = [im.size[0] for im in images]
        heights = [im.size[1] for im in images]
        total_w = sum(widths) + 10 * (len(images) - 1)
        H = max(heights) + 60
        canvas = Image.new("RGB", (total_w, H), (255, 255, 255))
        x = 0
        for j, (im, label) in enumerate(zip(images, labels)):
            canvas.paste(im, (x, 30))
            try:
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(canvas)
                font = ImageFont.load_default()
                color = "black" if j <= 1 else ("green" if "GT" in label else "black")
                draw.text((x, 5), label, fill=color, font=font)
            except Exception:
                pass
            x += im.size[0] + 10

        out_path = save_dir / f"error_{i}_query{rec['query_idx']}_{folder_name.replace('/', '_')}.jpg"
        canvas.save(str(out_path), quality=92)

    logging.info(f"错误样本图已保存到 {save_dir}，共 {min(len(error_indices), max_images)} 张（5 图并排：Query|Pred|GT_geo_nearest|GT_feat_best|GT_geo_farthest）")


def compare_baseline_improved(
    baseline_json: str,
    improved_json: str,
    output_dir: str,
    dataset_test_root: str,
    save_comparison_images: bool = True,
    max_per_category: int = 100,
):
    """
    对比 baseline 与 improved 的检索结果，输出 fixed / still wrong / new wrong。
    图像路径使用 JSON 内相对路径 + dataset_test_root 拼接。
    """
    with open(baseline_json, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    with open(improved_json, "r", encoding="utf-8") as f:
        improved_data = json.load(f)

    # 假设两条 JSON 都是 list of dict，且顺序一致（同一数据集同一 query 顺序）
    # 用 (query_folder, query_idx) 做 key
    def build_key(rec):
        return (rec["query_folder"], rec["query_idx"])

    baseline_by_key = {build_key(r): r for r in baseline_data["per_query_results"]}
    improved_by_key = {build_key(r): r for r in improved_data["per_query_results"]}

    fixed = []  # baseline 错 -> improved 对
    still_wrong = []  # 都错
    new_wrong = []  # baseline 对 -> improved 错

    for key, base in baseline_by_key.items():
        if key not in improved_by_key:
            continue
        imp = improved_by_key[key]
        base_wrong = not base["hit@1"]
        imp_wrong = not imp["hit@1"]
        if base_wrong and not imp_wrong:
            fixed.append((key, base, imp))
        elif base_wrong and imp_wrong:
            still_wrong.append((key, base, imp))
        elif not base_wrong and imp_wrong:
            new_wrong.append((key, base, imp))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "========== 改进前后对比报告 ==========",
        f"Baseline: {baseline_json}",
        f"Improved: {improved_json}",
        "",
        f"Fixed (原错→现对):     {len(fixed)}",
        f"Still wrong (仍错):    {len(still_wrong)}",
        f"New wrong (原对→现错): {len(new_wrong)}",
    ]
    report_path = out_path / "comparison_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    logging.info("\n".join(report_lines))

    # 保存分类 JSON
    summary = {
        "fixed": [{"query_folder": k[0], "query_idx": k[1]} for k, _, _ in fixed],
        "still_wrong": [{"query_folder": k[0], "query_idx": k[1]} for k, _, _ in still_wrong],
        "new_wrong": [{"query_folder": k[0], "query_idx": k[1]} for k, _, _ in new_wrong],
    }
    with open(out_path / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if not save_comparison_images:
        return

    def save_side_by_side(records: List, subdir: str, title_prefix: str):
        sub = out_path / subdir
        sub.mkdir(parents=True, exist_ok=True)
        for i, (key, base, imp) in enumerate(records[:max_per_category]):
            rec = imp  # 用 improved 的路径即可
            query_path = os.path.join(dataset_test_root, rec["query_image_path"])
            gt_path = os.path.join(dataset_test_root, rec["gt_image_path"]) if rec.get("gt_image_path") else ""
            pred_base = os.path.join(dataset_test_root, base["pred_image_path"]) if base.get("pred_image_path") else ""
            pred_imp = os.path.join(dataset_test_root, rec["pred_image_path"]) if rec.get("pred_image_path") else ""

            def load_resized(path, size=400):
                if not path or not os.path.isfile(path):
                    return Image.new("RGB", (size, size), (128, 128, 128))
                img = Image.open(path).convert("RGB")
                img.thumbnail((size, size * 2), Image.Resampling.LANCZOS)
                return img

            im_q = load_resized(query_path)
            im_gt = load_resized(gt_path)
            im_b = load_resized(pred_base)
            im_p = load_resized(pred_imp)
            W = im_q.size[0] + im_gt.size[0] + im_b.size[0] + im_p.size[0] + 30
            H = max(im_q.size[1], im_gt.size[1], im_b.size[1], im_p.size[1]) + 40
            canvas = Image.new("RGB", (W, H), (255, 255, 255))
            x = 0
            for label, img in [("Query", im_q), ("GT", im_gt), ("Baseline", im_b), ("Improved", im_p)]:
                canvas.paste(img, (x, 30))
                try:
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(canvas)
                    draw.text((x, 5), label, fill="black", font=ImageFont.load_default())
                except Exception:
                    pass
                x += img.size[0] + 10
            canvas.save(str(sub / f"{title_prefix}_{i}_q{key[1]}.jpg"), quality=92)

    if fixed:
        save_side_by_side(fixed, "fixed", "fixed")
    if still_wrong:
        save_side_by_side(still_wrong, "still_wrong", "still_wrong")
    if new_wrong:
        save_side_by_side(new_wrong, "new_wrong", "new_wrong")
    logging.info(f"对比图已保存到 {out_path}")


def main():
    ap = argparse.ArgumentParser(description="pitts30k 检索结果保存与改进前后对比")
    ap.add_argument("--resume", type=str, default=None, help="模型 checkpoint 路径")
    ap.add_argument("--eval_datasets_folder", type=str, default="/root/data/Pittsburgh")
    ap.add_argument("--eval_dataset_name", type=str, default="pitts30k")
    ap.add_argument("--save_dir", type=str, default="./logs/pitts30k_retrieval")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--output_json", type=str, default="retrieval_results.json")
    ap.add_argument("--save_error_images", action="store_true", help="保存 top1 错误样本并排图")
    ap.add_argument("--max_error_images", type=int, default=500)
    ap.add_argument("--comparison", action="store_true", help="仅运行对比模式")
    ap.add_argument("--baseline_json", type=str, default=None, help="对比：baseline 结果 JSON")
    ap.add_argument("--improved_json", type=str, default=None, help="对比：improved 结果 JSON")
    ap.add_argument("--comparison_dir", type=str, default=None, help="对比结果输出目录")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.save_dir, "retrieval_save_log.txt"), mode="a"),
        ],
    )

    if args.comparison and args.baseline_json and args.improved_json:
        dataset_test_root = os.path.join(args.eval_datasets_folder, args.eval_dataset_name, "test")
        out_dir = args.comparison_dir or os.path.join(args.save_dir, "comparison")
        compare_baseline_improved(
            args.baseline_json,
            args.improved_json,
            out_dir,
            dataset_test_root,
            save_comparison_images=True,
            max_per_category=100,
        )
        return

    # 构建与 eval 一致所需的 args（不调用项目 parser，避免吞掉本脚本的 --top_k 等参数）
    parser_args = argparse.Namespace(
        seed=42,
        device="cuda",
        num_workers=4,
        infer_batch_size=16,
        resize=[322, 322],
        test_method="hard_resize",
        majority_weight=0.01,
        val_positive_dist_threshold=25,
        recall_values=[1, 5, 10, 20],
        pca_dim=None,
        pca_dataset_folder=None,
        eval_datasets_folder=args.eval_datasets_folder,
        eval_dataset_name=args.eval_dataset_name,
        save_dir=args.save_dir,
        resume=args.resume or None,
    )

    commons.make_deterministic(parser_args.seed)
    model = network.VPRNet()
    model = model.to(parser_args.device)
    if parser_args.resume:
        import util
        model = util.resume_model(parser_args, model)
    model = torch.nn.DataParallel(model)
    parser_args.features_dim = 4096
    pca = None
    if getattr(parser_args, "pca_dim", None) is not None:
        import util
        pca = util.compute_pca(
            parser_args, model, parser_args.pca_dataset_folder, parser_args.features_dim
        )
        parser_args.features_dim = parser_args.pca_dim

    eval_ds = datasets_ws.BaseDataset(
        parser_args, parser_args.eval_datasets_folder, parser_args.eval_dataset_name, "test"
    )
    dataset_test_root = os.path.join(
        parser_args.eval_datasets_folder, parser_args.eval_dataset_name, "test"
    )

    logging.info("开始检索并收集 per-query 结果...")
    all_recalls, per_query_results = run_retrieval_and_collect_per_query(
        parser_args, model, eval_ds, pca, args.top_k, dataset_test_root
    )
    for folder_name, (_, recalls_str) in all_recalls.items():
        logging.info(f"Recalls {folder_name}: {recalls_str}")

    output_json_path = os.path.join(args.save_dir, args.output_json)
    output_data = {
        "dataset": parser_args.eval_dataset_name,
        "top_k": args.top_k,
        "recalls": {k: v[1] for k, v in all_recalls.items()},
        "per_query_results": per_query_results,
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logging.info(f"检索结果已保存到 {output_json_path}")

    # 筛选 top1 错误
    error_indices = [i for i, r in enumerate(per_query_results) if not r["hit@1"]]
    logging.info(f"Top1 错误样本数: {len(error_indices)}")

    if args.save_error_images and error_indices:
        save_error_sample_images(
            eval_ds,
            per_query_results,
            error_indices,
            Path(args.save_dir) / "error_samples_top1",
            dataset_test_root,
            max_images=args.max_error_images,
        )


if __name__ == "__main__":
    main()
