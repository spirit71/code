#!/usr/bin/env python3
"""构造 city-held-out GSV query-utility labels，并评估 train-only static ranking。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_query_retrieval_contribution import build_model
from src.analysis.reliability_diagnostics import descriptor_from_query_outputs


DEFAULT_VAL_CITIES = ["PRS", "Phoenix", "Rome", "TRT", "WashingtonDC"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gsv-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--places-per-city", type=int, default=128)
    parser.add_argument("--views-per-place", type=int, default=4)
    parser.add_argument("--loo-image-chunk", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-cities", nargs="*")
    parser.add_argument("--val-cities", nargs="*", default=DEFAULT_VAL_CITIES)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(value, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def stable_place_key(city: str, place_id: int, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{city}:{place_id}".encode()).hexdigest()


def image_name(row, place_id: int) -> str:
    local_place = str(int(place_id) % 10**5).zfill(7)
    return (
        f"{row['city_id']}_{local_place}_{str(row['year']).zfill(4)}_"
        f"{str(row['month']).zfill(2)}_{str(row['northdeg']).zfill(3)}_"
        f"{row['lat']}_{row['lon']}_{row['panoid']}.jpg"
    )


def select_city_images(root: Path, city: str, places: int, views: int, seed: int):
    frame = pd.read_csv(root / "Dataframes" / f"{city}.csv")
    counts = frame.groupby("place_id").size()
    eligible = [int(index) for index, count in counts.items() if count >= views]
    eligible.sort(key=lambda place: stable_place_key(city, place, seed))
    selected = eligible[:places]
    if len(selected) != places:
        raise RuntimeError(f"{city}: only {len(selected)} eligible places, need {places}")
    paths, labels, place_ids = [], [], []
    for local_label, place in enumerate(selected):
        rows = frame[frame["place_id"] == place].copy()
        rows["_name"] = [image_name(row, place) for _, row in rows.iterrows()]
        rows = rows.sort_values("_name").iloc[:views]
        for _, row in rows.iterrows():
            path = root / "Images" / str(row["city_id"]) / row["_name"]
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
            labels.append(local_label)
            place_ids.append(place)
    return paths, torch.tensor(labels, dtype=torch.long), place_ids


class ImageListDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = torchvision.io.decode_image(self.paths[index], mode="RGB")
        return self.transform(image), index


@torch.inference_mode()
def extract_city(model, paths, transform, args):
    loader = DataLoader(
        ImageListDataset(paths, transform), batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, shuffle=False,
    )
    descriptors, outputs = [], []
    expected_start = 0
    for images, indices in loader:
        expected = torch.arange(expected_start, expected_start + len(indices))
        if not torch.equal(indices, expected):
            raise RuntimeError("image order changed")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            descriptor, aux = model(images.to(args.device, non_blocking=True), return_aux=True)
        projection_input = aux["query_outputs_raw"]
        if descriptor.shape[1:] != (8192,) or projection_input.shape[1:] != (2, 64, 512):
            raise RuntimeError(
                f"unexpected descriptor/output shapes {descriptor.shape}/{projection_input.shape}"
            )
        descriptors.append(descriptor.float().cpu())
        outputs.append(projection_input.half().cpu())
        expected_start += len(indices)
    return torch.cat(descriptors), torch.cat(outputs)


def group_stats(similarity: torch.Tensor, labels: torch.Tensor, query_offset: int = 0):
    """计算 best positive、hard negative；query 自身始终排除。"""
    squeeze = similarity.ndim == 2
    if squeeze:
        similarity = similarity[:, None, :]
    batch, variants, references = similarity.shape
    best_positive = torch.empty(batch, variants, device=similarity.device)
    hard_negative = torch.empty_like(best_positive)
    work_positive = similarity.clone()
    work_negative = similarity.clone()
    all_labels = labels.to(similarity.device)
    for row in range(batch):
        global_index = query_offset + row
        positive = all_labels == all_labels[global_index]
        positive[global_index] = False
        negative = all_labels != all_labels[global_index]
        if int(positive.sum()) != 3 or int(negative.sum()) != references - 4:
            raise RuntimeError("positive/negative group cardinality failed")
        work_positive[row, :, ~positive] = -torch.inf
        work_negative[row, :, ~negative] = -torch.inf
    best_positive.copy_(work_positive.max(dim=-1).values)
    hard_negative.copy_(work_negative.max(dim=-1).values)
    if squeeze:
        return best_positive[:, 0], hard_negative[:, 0]
    return best_positive, hard_negative


@torch.inference_mode()
def compute_city_contribution(model, descriptors, outputs, labels, args):
    references = torch.nn.functional.normalize(descriptors.to(args.device), dim=-1)
    samples = len(outputs)
    contribution = torch.empty(samples, 64)
    masked_positive = torch.empty_like(contribution)
    masked_negative = torch.empty_like(contribution)
    baseline_positive = torch.empty(samples)
    baseline_negative = torch.empty(samples)
    for start in range(0, samples, args.loo_image_chunk):
        stop = min(start + args.loo_image_chunk, samples)
        source = outputs[start:stop].float().to(args.device)
        baseline_descriptor = descriptor_from_query_outputs(source, model.aggregator.fc)
        baseline_descriptor = torch.nn.functional.normalize(baseline_descriptor.float(), dim=-1)
        baseline_similarity = baseline_descriptor @ references.T
        positive, negative = group_stats(baseline_similarity, labels, start)
        current = stop - start
        expanded = source[:, None].expand(current, 64, 2, 64, 512).clone()
        diagonal = torch.arange(64, device=args.device)
        expanded[:, diagonal, -1, diagonal] = 0
        masked_descriptor = descriptor_from_query_outputs(
            expanded.reshape(current * 64, 2, 64, 512), model.aggregator.fc
        )
        masked_descriptor = torch.nn.functional.normalize(masked_descriptor.float(), dim=-1)
        masked_similarity = (masked_descriptor @ references.T).reshape(current, 64, samples)
        masked_pos, masked_neg = group_stats(masked_similarity, labels, start)
        value = (positive - negative)[:, None] - (masked_pos - masked_neg)
        identity = (
            value - ((positive[:, None] - masked_pos) - (negative[:, None] - masked_neg))
        ).abs().max()
        if float(identity) > 2e-6:
            raise RuntimeError(f"contribution identity failed: {float(identity)}")
        contribution[start:stop] = value.cpu()
        masked_positive[start:stop] = masked_pos.cpu()
        masked_negative[start:stop] = masked_neg.cpu()
        baseline_positive[start:stop] = positive.cpu()
        baseline_negative[start:stop] = negative.cpu()
    return {
        "contribution": contribution,
        "baseline_positive": baseline_positive,
        "baseline_negative": baseline_negative,
        "masked_positive": masked_positive,
        "masked_negative": masked_negative,
    }


def validate_city_cache(item, city, split, checkpoint_hash, expected_images):
    if (
        item["city"] != city or item["split"] != split
        or item["checkpoint_sha256"] != checkpoint_hash
        or item["projection_inputs"].shape != (expected_images, 2, 64, 512)
        or item["contribution"].shape != (expected_images, 64)
    ):
        raise RuntimeError(f"invalid city cache: {city}")


@torch.inference_mode()
def masked_group_recall(model, item, remove_indices, args):
    outputs = item["projection_inputs"].float().to(args.device)
    if remove_indices:
        outputs[:, -1, remove_indices] = 0
    queries = descriptor_from_query_outputs(outputs, model.aggregator.fc).float()
    references = item["descriptors"].float().to(args.device)
    distance = (
        (queries * queries).sum(1, keepdim=True)
        + (references * references).sum(1)[None]
        - 2 * queries @ references.T
    )
    diagonal = torch.arange(len(queries), device=args.device)
    distance[diagonal, diagonal] = torch.inf
    prediction = distance.topk(20, largest=False).indices.cpu()
    labels = item["labels"]
    result = {}
    for k in (1, 5, 10, 20):
        result[k] = float(
            (labels[prediction[:, :k]] == labels[:, None]).any(dim=1).float().mean()
        )
    return result


def main():
    args = parse_args()
    if args.views_per_place != 4:
        raise ValueError("current positive-cardinality checks require exactly 4 views/place")
    if args.smoke:
        args.places_per_city = min(args.places_per_city, 8)
        args.train_cities = ["Bangkok"]
        args.val_cities = ["WashingtonDC"]
    all_cities = sorted(path.stem for path in (args.gsv_root / "Dataframes").glob("*.csv"))
    val_cities = list(args.val_cities)
    train_cities = list(args.train_cities) if args.train_cities else [
        city for city in all_cities if city not in val_cities
    ]
    if set(train_cities) & set(val_cities):
        raise RuntimeError("train/validation city overlap")
    if sorted(train_cities + val_cities) != all_cities and not args.smoke:
        raise RuntimeError("full run must partition all available cities exactly")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    city_dir = args.output_dir / "cities"
    city_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    checkpoint_hash = sha256_file(args.checkpoint)
    model = build_model(args.checkpoint, args.device, "e0", "off")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen E0 contains trainable parameter")
    transform = T.Compose([
        T.Resize((322, 322), interpolation=3),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    city_items = {"train": [], "val": []}
    for split, cities in (("train", train_cities), ("val", val_cities)):
        for city in cities:
            path = city_dir / f"{split}_{city}.pt"
            expected_images = args.places_per_city * args.views_per_place
            if path.is_file():
                item = torch.load(path, map_location="cpu", weights_only=True)
                validate_city_cache(item, city, split, checkpoint_hash, expected_images)
                print(f"[cache] {path.name}", flush=True)
            else:
                paths, labels, place_ids = select_city_images(
                    args.gsv_root, city, args.places_per_city,
                    args.views_per_place, args.seed,
                )
                descriptors, outputs = extract_city(model, paths, transform, args)
                reconstructed = descriptor_from_query_outputs(
                    outputs.float().to(args.device), model.aggregator.fc
                ).cpu()
                minimum_cosine = float(torch.nn.functional.cosine_similarity(
                    reconstructed, descriptors
                ).min())
                if minimum_cosine < 0.9999:
                    raise RuntimeError(f"{city} projection fidelity {minimum_cosine}")
                stats = compute_city_contribution(model, descriptors, outputs, labels, args)
                item = {
                    "format_version": 1, "split": split, "city": city,
                    "checkpoint": str(args.checkpoint.resolve()),
                    "checkpoint_sha256": checkpoint_hash,
                    "places": args.places_per_city, "views_per_place": args.views_per_place,
                    "image_paths": [str(value) for value in paths],
                    "place_ids": place_ids, "labels": labels,
                    # Static validation 使用正式 L2 排名，descriptor 保留 FP32。
                    "descriptors": descriptors,
                    "projection_inputs": outputs,
                    "projection_minimum_cosine": minimum_cosine,
                    **stats,
                }
                atomic_save(item, path)
                print(f"[saved] {path.name}", flush=True)
            city_items[split].append(item)

    train_contribution = torch.cat([item["contribution"] for item in city_items["train"]])
    val_contribution = torch.cat([item["contribution"] for item in city_items["val"]])
    mean_contribution = train_contribution.mean(dim=0)
    static_order = torch.argsort(mean_contribution).tolist()
    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
    fc_norm = state["aggregator.fc.weight"][:, 64:128].float().norm(dim=0)
    fc_order = torch.argsort(fc_norm).tolist()
    mask_counts = [0, 4, 8, 16, 24, 32]
    rows = []
    for method, order in (("train_mean_contribution", static_order), ("fc_slot_norm", fc_order)):
        for count in mask_counts:
            aggregate = {1: 0.0, 5: 0.0, 10: 0.0, 20: 0.0}
            total = 0
            for item in city_items["val"]:
                recall = masked_group_recall(model, item, order[:count], args)
                images = len(item["labels"])
                total += images
                for k in aggregate:
                    aggregate[k] += recall[k] * images
            rows.append({
                "split": "city_heldout_val", "method": method, "removed": count,
                **{f"R@{k}": aggregate[k] / total for k in aggregate},
            })
    train_rows = [row for row in rows if row["method"] == "train_mean_contribution"]
    best = max(train_rows, key=lambda row: (row["R@1"], -row["removed"]))
    elapsed = time.perf_counter() - started
    with (args.output_dir / "validation_static_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "query_index_train_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "query_index", "mean_train_contribution", "mean_val_contribution",
            "train_negative_ratio", "val_negative_ratio", "fc_slot_norm",
            "train_rank_low_to_high",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        rank = torch.argsort(torch.argsort(mean_contribution))
        for index in range(64):
            writer.writerow({
                "query_index": index,
                "mean_train_contribution": float(mean_contribution[index]),
                "mean_val_contribution": float(val_contribution[:, index].mean()),
                "train_negative_ratio": float((train_contribution[:, index] < 0).float().mean()),
                "val_negative_ratio": float((val_contribution[:, index] < 0).float().mean()),
                "fc_slot_norm": float(fc_norm[index]),
                "train_rank_low_to_high": int(rank[index]),
            })
    summary = {
        "experiment": "GSV city-held-out utility labels and static baseline",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "train_cities": train_cities, "val_cities": val_cities,
        "places_per_city": args.places_per_city,
        "views_per_place": args.views_per_place,
        "train_images": len(train_contribution), "val_images": len(val_contribution),
        "input_size": [322, 322],
        "mining_group": (
            f"one city; {args.places_per_city} places x {args.views_per_place} views; "
            "best positive and batch hard negative"
        ),
        "train_contribution": {
            "mean": float(train_contribution.mean()),
            "std": float(train_contribution.std(unbiased=False)),
            "negative_fraction": float((train_contribution < 0).float().mean()),
        },
        "val_contribution": {
            "mean": float(val_contribution.mean()),
            "std": float(val_contribution.std(unbiased=False)),
            "negative_fraction": float((val_contribution < 0).float().mean()),
        },
        "train_static_order_low_to_high": static_order,
        "fc_norm_order_low_to_high": fc_order,
        "selected_removed_by_val_r1": best["removed"],
        "selected_val_recall": {k: best[k] for k in ("R@1", "R@5", "R@10", "R@20")},
        "validation_rows": rows,
        "all_parameters_frozen": True,
        "test_gt_used": False,
        "deployable_static_ranking": True,
        "smoke": args.smoke,
        "elapsed_seconds": elapsed,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "checkpoint_hash.txt").write_text(
        checkpoint_hash + "\n", encoding="utf-8"
    )
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps({
            "checkpoint": str(args.checkpoint.resolve()),
            "gsv_root": str(args.gsv_root.resolve()),
            "train_cities": train_cities,
            "val_cities": val_cities,
            "places_per_city": args.places_per_city,
            "views_per_place": args.views_per_place,
            "input_size": [322, 322],
            "batch_size": args.batch_size,
            "loo_image_chunk": args.loo_image_chunk,
            "seed": args.seed,
            "mask_counts": mask_counts,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# GSV City-held-out Utility Labels 与 Static Baseline", "",
        f"- train/val images: `{len(train_contribution)}/{len(val_contribution)}`",
        f"- selected removed by val R@1: `{best['removed']}`",
        f"- elapsed: `{elapsed/60:.2f} min`", "",
        "| Method | Removed | R@1 | R@5 | R@10 | R@20 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['removed']} | {100*row['R@1']:.2f} | "
            f"{100*row['R@5']:.2f} | {100*row['R@10']:.2f} | {100*row['R@20']:.2f} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"complete: {args.output_dir} ({elapsed/60:.2f} min)", flush=True)


if __name__ == "__main__":
    main()
