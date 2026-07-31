#!/usr/bin/env python3
"""按原 C8/C9 代码确定性重建并保存 MSLS cross-city fold models。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_msls_cross_city_ranknet import candidate_labels, train_one_epoch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utility-cache-root", type=Path, required=True)
    parser.add_argument("--candidate-cache-root", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    return parser.parse_args()


def main():
    args = parse_args()
    utility = torch.load(
        args.utility_cache_root / "msls_val.pt",
        map_location="cpu", weights_only=True,
    )
    candidate = torch.load(
        args.candidate_cache_root / "msls_c0.pt",
        map_location="cpu", weights_only=True,
    )
    feature_item = torch.load(args.feature_cache, map_location="cpu", weights_only=True)
    features = feature_item["val"].float()
    labels = candidate_labels(candidate["candidate_indices"], utility["ground_truth"])
    cities = [value.split("/")[0] for value in utility["image_ids"]]
    city_rows = {
        city: torch.tensor([i for i, value in enumerate(cities) if value == city])
        for city in sorted(set(cities))
    }
    if {key: len(value) for key, value in city_rows.items()} != {"cph": 498, "sf": 242}:
        raise RuntimeError("unexpected city split")
    metadata = []
    for label, epochs in (("c8_e1", 1), ("c9_e20", 20)):
        output = args.output_root / label
        output.mkdir(parents=True, exist_ok=True)
        train_args = SimpleNamespace(
            device=args.device, epochs=epochs, batch_size=256,
            learning_rate=1e-3, weight_decay=1e-4,
        )
        for seed in args.seeds:
            for train_city in ("cph", "sf"):
                path = output / f"seed_{seed}_train_{train_city}.pt"
                if path.is_file():
                    item = torch.load(path, map_location="cpu", weights_only=True)
                    if item["epochs"] != epochs or item["seed"] != seed:
                        raise RuntimeError(f"invalid existing fold model {path}")
                    metadata.append({key: value for key, value in item.items() if key != "state_dict"})
                    continue
                model, loss, pairs, used = train_one_epoch(
                    features, labels, city_rows[train_city],
                    list(range(7)), seed, train_args,
                )
                item = {
                    "architecture": "CandidateRankNet(7,16,1)",
                    "feature_names": feature_item["feature_names"],
                    "source_experiment": label,
                    "epochs": epochs, "seed": seed, "train_city": train_city,
                    "train_queries": used, "train_loss": loss, "train_pairs": pairs,
                    "state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                }
                torch.save(item, path)
                metadata.append({key: value for key, value in item.items() if key != "state_dict"})
                print(json.dumps(metadata[-1]), flush=True)
    (args.output_root / "summary.json").write_text(json.dumps({
        "protocol": "deterministic reconstruction of C8/C9 cross-city fold models",
        "models": metadata,
        "test_data_used": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
