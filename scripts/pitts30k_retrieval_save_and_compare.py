#!/usr/bin/env python3
"""
2026年3月7日 16点19分
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
    # 用于生成相对路径的 dataset test 根目录
    dataset_test_root = os.path.join(
        args.eval_datasets_folder, args.eval_dataset_name, "test"
    )

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
            gt_indices = positives_per_query[query_index]
            retrieved = pred[:top_k].tolist()
            dist_topk = distances[query_index][:top_k].tolist()
            scores = [round(l2_distance_to_similarity(d), 4) for d in dist_topk]
            hit1 = int(pred[0]) in gt_indices
            hit5 = any(int(p) in gt_indices for p in pred[:5])
            hit10 = any(int(p) in gt_indices for p in pred[: min(10, len(pred))])
            hit20 = any(int(p) in gt_indices for p in pred[: min(20, len(pred))])

            query_path = query_paths[query_index]
            gt_path = database_paths[gt_indices[0]] if len(gt_indices) > 0 else ""
            pred_path = database_paths[int(pred[0])] if len(pred) > 0 else ""

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
                "gt_image_path": make_relative_path(gt_path) if gt_path else "",
                "pred_image_path": make_relative_path(pred_path) if pred_path else "",
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
    将 top1 错误样本保存为并排图：query | gt | pred。
    error_indices: 在 per_query_results 中的下标（仅 top1 错误的记录）。
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    images_paths = eval_ds.images_paths
    database_paths = eval_ds.database_paths
    database_num = eval_ds.database_num

    def load_resized(path: str, size: Tuple[int, int]) -> Image.Image:
        if not path or not os.path.isfile(path):
            img = Image.new("RGB", size, (128, 128, 128))
            return img
        img = Image.open(path).convert("RGB")
        img.thumbnail((composite_width, composite_width * 2), Image.Resampling.LANCZOS)
        return img

    for i, rec_idx in enumerate(error_indices[:max_images]):
        rec = per_query_results[rec_idx]
        folder_name = rec["query_folder"]
        query_path = rec.get("query_image_path") or rec["query_path"]
        gt_path = rec.get("gt_image_path") or ""
        pred_path = rec.get("pred_image_path") or ""

        # 解析为绝对路径
        if not os.path.isabs(query_path):
            query_path = os.path.join(dataset_test_root, query_path)
        if gt_path and not os.path.isabs(gt_path):
            gt_path = os.path.join(dataset_test_root, gt_path)
        if pred_path and not os.path.isabs(pred_path):
            pred_path = os.path.join(dataset_test_root, pred_path)

        im_q = load_resized(query_path, (composite_width, composite_width))
        im_gt = load_resized(gt_path, (composite_width, composite_width))
        im_pred = load_resized(pred_path, (composite_width, composite_width))

        w1, h1 = im_q.size
        w2, h2 = im_gt.size
        w3, h3 = im_pred.size
        H = max(h1, h2, h3)
        canvas = Image.new("RGB", (w1 + w2 + w3 + 20, H + 60), (255, 255, 255))
        canvas.paste(im_q, (0, 30))
        canvas.paste(im_gt, (w1 + 10, 30))
        canvas.paste(im_pred, (w1 + w2 + 20, 30))

        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default()
            draw.text((0, 5), "Query", fill="black", font=font)
            draw.text((w1 + 10, 5), "GT", fill="green", font=font)
            draw.text((w1 + w2 + 20, 5), "Pred", fill="red", font=font)
        except Exception:
            pass

        out_path = save_dir / f"error_{i}_query{rec['query_idx']}_{folder_name.replace('/', '_')}.jpg"
        canvas.save(str(out_path), quality=92)

    logging.info(f"错误样本图已保存到 {save_dir}，共 {min(len(error_indices), max_images)} 张")


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
