from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Any, Tuple

import numpy as np
import torchvision
from scipy.spatial import cKDTree
from torch.utils.data import Dataset


class PairedNameTestDataset(Dataset):
    # 适用于“query 图片名 == reference 图片名”这种一一精确配对的数据集。
    """测试集: query 与同名 reference 互为正样本。"""

    def __init__(
        self,
        dataset_path: str | Path,
        dataset_name: str,
        reference_subdir: str = "test/database",
        query_subdir: str = "test/queries",
        transform: Optional[Callable] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset_name
        self.transform = transform

        self.reference_dir = self.dataset_path / reference_subdir
        self.query_dir = self.dataset_path / query_subdir
        if not self.reference_dir.is_dir():
            raise FileNotFoundError(f"Reference directory not found: {self.reference_dir}")
        if not self.query_dir.is_dir():
            raise FileNotFoundError(f"Query directory not found: {self.query_dir}")

        self.dbImages = sorted(path.relative_to(self.dataset_path).as_posix() for path in self.reference_dir.iterdir() if path.is_file())
        self.qImages = sorted(path.relative_to(self.dataset_path).as_posix() for path in self.query_dir.iterdir() if path.is_file())

        # 建立 文件名 -> reference 索引 的字典，后面每个 query 都能快速找到自己的唯一正样本。
        ref_index = {Path(path).name: idx for idx, path in enumerate(self.dbImages)}
        self.ground_truth = []
        for query_path in self.qImages:
            query_name = Path(query_path).name
            if query_name not in ref_index:
                raise ValueError(f"No reference image found for query {query_name} in {self.dataset_name}")
            self.ground_truth.append(np.array([ref_index[query_name]], dtype=np.int64))

        self.image_paths = self.dbImages + self.qImages
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        img = torchvision.io.decode_image(self.dataset_path / self.image_paths[index], mode="RGB")
        if self.transform:
            img = self.transform(img)
        return img, index

    def __len__(self) -> int:
        return len(self.image_paths)


class NordlandSequenceDataset(Dataset):
    # Nordland 不是“同名精确配对”，而是“序列附近若干帧都算正确”。
    # 这里按官方常见设定实现为 +/- 10 frame 容忍窗口。
    """测试集: query i 与 reference 中 [i-10, i+10] 的帧都算正样本。"""

    def __init__(
        self,
        dataset_path: str | Path,
        dataset_name: str = "nordland",
        reference_subdir: str = "test/database",
        query_subdir: str = "test/queries",
        frame_tolerance: int = 10,
        transform: Optional[Callable] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset_name
        self.transform = transform
        self.frame_tolerance = frame_tolerance

        self.reference_dir = self.dataset_path / reference_subdir
        self.query_dir = self.dataset_path / query_subdir
        if not self.reference_dir.is_dir():
            raise FileNotFoundError(f"Reference directory not found: {self.reference_dir}")
        if not self.query_dir.is_dir():
            raise FileNotFoundError(f"Query directory not found: {self.query_dir}")

        self.dbImages = sorted(path.relative_to(self.dataset_path).as_posix() for path in self.reference_dir.iterdir() if path.is_file())
        self.qImages = sorted(path.relative_to(self.dataset_path).as_posix() for path in self.query_dir.iterdir() if path.is_file())

        # Nordland 的 summer/winter 序列必须等长，否则 frame-to-frame 对齐逻辑就不成立。
        if len(self.dbImages) != len(self.qImages):
            raise ValueError(
                f"Nordland sequence evaluation expects equal sequence lengths, got {len(self.dbImages)} references and {len(self.qImages)} queries"
            )

        # 对第 i 个 query，reference 中 [i - tolerance, i + tolerance] 范围都算正确检索结果。
        self.ground_truth = []
        num_frames = len(self.dbImages)
        for query_idx in range(num_frames):
            start = max(0, query_idx - self.frame_tolerance)
            end = min(num_frames, query_idx + self.frame_tolerance + 1)
            self.ground_truth.append(np.arange(start, end, dtype=np.int64))

        self.image_paths = self.dbImages + self.qImages
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        img = torchvision.io.decode_image(self.dataset_path / self.image_paths[index], mode="RGB")
        if self.transform:
            img = self.transform(img)
        return img, index

    def __len__(self) -> int:
        return len(self.image_paths)


class CoordinateRadiusTestDataset(Dataset):
    # 适用于“按地理坐标半径定义正样本”的数据集，
    # 如 Pitts30k / Tokyo247 / SVOX 这类带 UTM 坐标的 benchmark。
    """测试集: 距离 query 一定半径内的 reference 都算正样本。"""

    def __init__(
        self,
        dataset_path: str | Path,
        dataset_name: str,
        radius_m: float,
        reference_subdir: str,
        query_subdirs: list[str],
        transform: Optional[Callable] = None,
        drop_queries_without_positives: bool = False,
    ):
        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset_name
        self.radius_m = radius_m
        self.transform = transform
        self.drop_queries_without_positives = drop_queries_without_positives

        self.reference_dir = self.dataset_path / reference_subdir
        if not self.reference_dir.is_dir():
            raise FileNotFoundError(f"Reference directory not found: {self.reference_dir}")

        self.query_dirs = [self.dataset_path / subdir for subdir in query_subdirs]
        missing = [str(path) for path in self.query_dirs if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"Query directories not found for {self.dataset_name}: {missing}")

        self.dbImages = sorted(path.relative_to(self.dataset_path).as_posix() for path in self.reference_dir.iterdir() if path.is_file())
        self.qImages = []
        for query_dir in self.query_dirs:
            self.qImages.extend(sorted(path.relative_to(self.dataset_path).as_posix() for path in query_dir.iterdir() if path.is_file()))

        # 从文件名里解析 UTM 风格坐标，后面根据半径自动构 ground truth。
        self.db_coords = np.stack([self._parse_coords(path) for path in self.dbImages], axis=0)
        self.q_coords = np.stack([self._parse_coords(path) for path in self.qImages], axis=0)

        # 用 KD-tree 做“半径内找所有 reference”的检索，速度比暴力两两算距离快很多。
        tree = cKDTree(self.db_coords)
        raw_ground_truth = tree.query_ball_point(self.q_coords, r=self.radius_m)
        empty_queries = [idx for idx, positives in enumerate(raw_ground_truth) if len(positives) == 0]
        if empty_queries and not self.drop_queries_without_positives:
            raise ValueError(
                f"{self.dataset_name} has {len(empty_queries)} queries without positives within {self.radius_m} m"
            )
        # 某些本地数据拷贝可能有极少数 query 在给定半径内没有正样本。
        # 对 SVOX 这类情况，允许把这些 query 丢掉，避免整个流程直接报错。
        if empty_queries and self.drop_queries_without_positives:
            keep_mask = np.array([len(positives) > 0 for positives in raw_ground_truth], dtype=bool)
            self.qImages = [path for path, keep in zip(self.qImages, keep_mask) if keep]
            self.q_coords = self.q_coords[keep_mask]
            raw_ground_truth = [positives for positives in raw_ground_truth if len(positives) > 0]

        self.ground_truth = [np.array(positives, dtype=np.int64) for positives in raw_ground_truth]

        self.image_paths = self.dbImages + self.qImages
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    @staticmethod
    def _parse_coords(relative_path: str) -> np.ndarray:
        # 文件名格式类似: xxx@easting@northing@...
        # 这里取第 1、2 段作为平面坐标。
        name = Path(relative_path).name
        parts = name.split('@')
        if len(parts) < 3:
            raise ValueError(f"Cannot parse coordinates from filename: {name}")
        return np.array([float(parts[1]), float(parts[2])], dtype=np.float32)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        img = torchvision.io.decode_image(self.dataset_path / self.image_paths[index], mode="RGB")
        if self.transform:
            img = self.transform(img)
        return img, index

    def __len__(self) -> int:
        return len(self.image_paths)
