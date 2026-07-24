from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import shutil
from pathlib import Path
import time

import numpy as np
import torch


def checkpoint_sha256(path: str | Path) -> str:
    """计算 checkpoint 内容指纹，防止误用其他权重生成的旧缓存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class BoQFeatureBundle:
    """一个数据集的全局特征、局部特征和注意力的统一容器。

    数据必须严格按 ``references 在前、queries 在后`` 排列；Recall 计算依赖
    这个顺序，不能只看 dataloader 返回顺序就直接拼接。
    """
    global_descriptors: torch.Tensor
    local_features: dict[str, torch.Tensor]
    attentions: torch.Tensor | None
    indices: torch.Tensor
    num_references: int
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        total = self.global_descriptors.shape[0]
        if self.indices.shape[0] != total or not torch.equal(self.indices, torch.arange(total)):
            raise ValueError("Feature bundle must be restored to contiguous dataset-index order")
        if not 0 < self.num_references < total:
            raise ValueError("Invalid reference/query split")
        for name, value in self.local_features.items():
            if value.shape[0] != total:
                raise ValueError(f"Local feature {name} has inconsistent image count")

    @property
    def num_queries(self):
        return self.global_descriptors.shape[0] - self.num_references

    def split(self, tensor: torch.Tensor):
        """按数据集规定的位置切成 reference 部分和 query 部分。"""
        return tensor[:self.num_references], tensor[self.num_references:]

    def layer(self, name: str):
        return self.split(self.local_features[name])

    @property
    def global_split(self):
        return self.split(self.global_descriptors)

    def bytes_per_image(self) -> float:
        tensors = [self.global_descriptors, *self.local_features.values()]
        if self.attentions is not None:
            tensors.append(self.attentions)
        return sum(t.numel() * t.element_size() for t in tensors) / len(self.indices)


class FeatureCache:
    """磁盘特征缓存：避免每个消融实验都重新跑一次大模型。"""
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(metadata: dict) -> str:
        # 把 checkpoint、数据集、输入尺寸、所需层等信息编码成唯一目录名。
        base = {k: v for k, v in metadata.items() if k not in {"spatial_shape", "token_count", "feature_dim", "dtype"}}
        canonical = json.dumps(base, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:20]

    def path(self, metadata: dict):
        return self.root / f"{metadata['dataset']}__{self.key(metadata)}.pt"

    def directory(self, metadata: dict):
        return self.root / f"{metadata['dataset']}__{self.key(metadata)}"

    def load(self, metadata: dict) -> BoQFeatureBundle | None:
        """元数据完全一致才命中缓存；任何关键条件变化都会重新提特征。"""
        directory = self.directory(metadata)
        manifest_path = directory / "metadata.json"
        if manifest_path.exists():
            cached_metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            if any(cached_metadata.get(k) != value for k, value in metadata.items()):
                return None
            tensor_names = cached_metadata["cache_tensor_names"]
            # mmap 只在访问某段数据时读取对应磁盘页，避免大数据集一次塞满内存。
            tensors = {
                name: torch.from_numpy(np.load(directory / f"{name}.npy", mmap_mode="r+"))
                for name in tensor_names
            }
            total = tensors["global"].shape[0]
            return BoQFeatureBundle(
                global_descriptors=tensors.pop("global"),
                local_features={k: v for k, v in tensors.items() if k != "attention"},
                attentions=tensors.get("attention"),
                indices=torch.arange(total),
                num_references=cached_metadata["num_references"],
                metadata=cached_metadata,
            )
        path = self.path(metadata)
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if any(payload["metadata"].get(k) != value for k, value in metadata.items()):
            return None
        return BoQFeatureBundle(**payload["bundle"])

    def save(self, bundle: BoQFeatureBundle):
        path = self.path(bundle.metadata)
        torch.save({"metadata": bundle.metadata, "bundle": bundle.__dict__}, path)
        return path


def make_cache_metadata(checkpoint, dataset, input_size, model, requested_layers):
    """记录影响特征内容的条件，用于判断缓存能否安全复用。"""
    return {
        "schema_version": 1,
        "checkpoint_hash": checkpoint_sha256(checkpoint),
        "dataset": dataset.dataset_name,
        "dataset_length": len(dataset),
        "num_references": dataset.num_references,
        "input_size": list(input_size),
        "backbone": getattr(model.backbone, "backbone_name", type(model.backbone).__name__),
        "requested_layers": list(requested_layers),
        "local_dtype": "float16",
        "global_dtype": "float32",
    }


def extract_feature_bundle(model, dataloader, device, metadata: dict, cache: FeatureCache | None = None):
    """遍历整个数据集，提取 global、指定局部层和最后一层 attention。

    局部特征/attention 以 float16 保存以节约空间；最终用于 FAISS 的 global
    保持 float32。返回值还会说明是否命中缓存和本次提取耗时。
    """
    if cache:
        hit = cache.load(metadata)
        if hit is not None:
            return hit, {"cache_hit": True, "seconds": 0.0}
    # eval + inference_mode：关闭 dropout 和梯度，保证推理稳定并节省显存。
    model.eval()
    chunks, index_chunks = {}, []
    cache_directory = cache.directory(metadata) if cache else None
    # 先写 .partial；全部成功后再改名。中途崩溃不会留下看似完整的缓存。
    partial_directory = cache_directory.with_name(cache_directory.name + ".partial") if cache_directory else None
    disk_arrays = None
    if partial_directory is not None:
        if partial_directory.exists():
            shutil.rmtree(partial_directory)
        partial_directory.mkdir(parents=True)
    start = time.perf_counter()
    with torch.inference_mode():
        total_batches = len(dataloader)
        for batch_idx, (images, indices) in enumerate(dataloader):
            # 一次前向同时取得全局描述符和 QASSR 所需的中间张量。
            result = model(images.to(device, non_blocking=True), return_intermediates=True)
            available = {"global": result["global"], "raw_backbone": result["raw_backbone"], "x0": result["x0"]}
            # encoder_outputs[0]/[1] 分别对外命名为 x1/x2。
            available.update({f"x{i+1}": value for i, value in enumerate(result["encoder_outputs"])})
            available["attention"] = result["last_attention"]
            requested = {"global", *metadata.get("requested_layers", [])}
            values = {name: value for name, value in available.items() if name in requested}
            if disk_arrays is None and partial_directory is not None:
                disk_arrays = {}
                for name, value in values.items():
                    if value is not None:
                        dtype = np.float32 if name == "global" else np.float16
                        shape = (len(dataloader.dataset), *value.shape[1:])
                        disk_arrays[name] = np.lib.format.open_memmap(partial_directory / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
            # dataloader 可能打乱批次；保存原始数据集 index，最后再恢复正确顺序。
            cpu_indices = indices.detach().cpu().long()
            for name, value in values.items():
                if value is None:
                    continue
                dtype = torch.float32 if name == "global" else torch.float16
                cpu_value = value.detach().to("cpu", dtype=dtype)
                if disk_arrays is not None:
                    disk_arrays[name][cpu_indices.numpy()] = cpu_value.numpy()
                else:
                    chunks.setdefault(name, []).append(cpu_value)
            index_chunks.append(cpu_indices)
            metadata.setdefault("spatial_shape", list(result["spatial_shape"]))
            if batch_idx == 0 or (batch_idx + 1) % 50 == 0 or batch_idx + 1 == total_batches:
                print(
                    f"[feature-extraction] dataset={dataloader.dataset.dataset_name} "
                    f"batch={batch_idx + 1}/{total_batches} "
                    f"images={min((batch_idx + 1) * dataloader.batch_size, len(dataloader.dataset))}/{len(dataloader.dataset)}",
                    flush=True,
                )
    indices = torch.cat(index_chunks)
    order = indices.argsort()
    if disk_arrays is not None:
        if not torch.equal(indices[order], torch.arange(len(dataloader.dataset))):
            raise RuntimeError("Feature extraction did not cover every dataset index exactly once")
        for array in disk_arrays.values():
            array.flush()
        ordered = {name: torch.from_numpy(array) for name, array in disk_arrays.items()}
    else:
        ordered = {name: torch.cat(parts)[order] for name, parts in chunks.items()}
    metadata.update({
        "token_count": {k: int(v.shape[1]) for k, v in ordered.items() if v.ndim == 3},
        "feature_dim": {k: int(v.shape[-1]) for k, v in ordered.items() if v.ndim >= 2},
        "dtype": {k: str(v.dtype).replace("torch.", "") for k, v in ordered.items()},
        "cache_format": "npy_memmap_v1" if disk_arrays is not None else "torch_pt_v1",
        "cache_tensor_names": list(ordered),
    })
    bundle = BoQFeatureBundle(
        global_descriptors=ordered.pop("global"), local_features={k: v for k, v in ordered.items() if k != "attention"},
        attentions=ordered.get("attention"), indices=indices[order], num_references=dataloader.dataset.num_references,
        metadata=metadata,
    )
    if cache and disk_arrays is not None:
        (partial_directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if cache_directory.exists():
            shutil.rmtree(cache_directory)
        partial_directory.rename(cache_directory)
    elif cache:
        cache.save(bundle)
    return bundle, {"cache_hit": False, "seconds": time.perf_counter() - start}
