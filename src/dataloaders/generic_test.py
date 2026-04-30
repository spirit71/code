# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

"""Generic VPR test dataset with UTM-based soft positive mining."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F


class GenericVPRTestDataset(Dataset):
    """
    Generic test dataset for VPR evaluation.

    The expected folder layout is:
    `dataset_path / split / {"database", "queries"}` containing `.jpg` files.

    Key interfaces aligned with BoQ eval flow:
    - `dataset_name`
    - `num_references`
    - `num_queries`
    - `ground_truth`
    - `__getitem__` returns `(tensor, index)`
    """

    SUPPORTED_TEST_METHODS = ("hard_resize", "central_crop", "five_crops")

    def __init__(
        self,
        dataset_path: str | Path,
        split: str = "test",
        test_method: str = "hard_resize",
        image_size: Tuple[int, int] = (322, 322),
        positive_dist_threshold: float = 25.0,
        dataset_name: Optional[str] = None,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.test_method = test_method
        self.image_size = image_size
        self.positive_dist_threshold = positive_dist_threshold
        self.dataset_name = dataset_name or self.dataset_path.name
        self.mean = list(mean)
        self.std = list(std)

        self._validate_inputs()
        self.database_dir, self.queries_dir = self._resolve_split_dirs()

        self.db_image_paths = self._scan_jpg(self.database_dir)
        self.q_image_paths = self._scan_jpg(self.queries_dir)
        self.image_paths = self.db_image_paths + self.q_image_paths

        self.num_references = len(self.db_image_paths)
        self.num_queries = len(self.q_image_paths)

        self.db_coords = np.asarray(
            [self._parse_utm_coords(path) for path in self.db_image_paths], dtype=np.float64
        )
        self.q_coords = np.asarray(
            [self._parse_utm_coords(path) for path in self.q_image_paths], dtype=np.float64
        )

        self.ground_truth = self._compute_soft_positives()
        self.normalize = T.Normalize(mean=self.mean, std=self.std)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        image_path = self.image_paths[index]
        image = torchvision.io.decode_image(str(image_path), mode="RGB")
        image = self._apply_test_method(image)
        return image, index

    def get_positives(self) -> List[np.ndarray]:
        """Return soft positives for each query as database indices."""
        return self.ground_truth

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dataset_name={self.dataset_name!r}, split={self.split!r}, "
            f"test_method={self.test_method!r}, image_size={self.image_size}, "
            f"num_references={self.num_references}, num_queries={self.num_queries}, "
            f"positive_dist_threshold={self.positive_dist_threshold})"
        )

    def _validate_inputs(self) -> None:
        if self.test_method not in self.SUPPORTED_TEST_METHODS:
            raise ValueError(
                f"Unknown test_method={self.test_method!r}. "
                f"Supported methods: {self.SUPPORTED_TEST_METHODS}"
            )

        if not self.dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        has_nested_split = self._has_valid_split(self.dataset_path / self.split)
        has_direct_split = self._has_valid_split(self.dataset_path)
        if not has_nested_split and not has_direct_split:
            raise FileNotFoundError(
                "Expected folders not found. Supported layouts are either "
                "`dataset_path/split/{database|gallery} + queries` or "
                "`dataset_path/{database|gallery} + queries`."
            )

    def _resolve_split_dirs(self) -> Tuple[Path, Path]:
        nested_dirs = self._get_split_dirs(self.dataset_path / self.split)
        if nested_dirs is not None:
            return nested_dirs
        return self._get_split_dirs(self.dataset_path)  # type: ignore[return-value]

    @staticmethod
    def _has_valid_split(base_dir: Path) -> bool:
        if not base_dir.is_dir():
            return False
        has_db = (base_dir / "database").is_dir() or (base_dir / "gallery").is_dir()
        has_queries = (base_dir / "queries").is_dir()
        return has_db and has_queries

    @staticmethod
    def _get_split_dirs(base_dir: Path) -> Optional[Tuple[Path, Path]]:
        if not base_dir.is_dir():
            return None
        queries_dir = base_dir / "queries"
        if not queries_dir.is_dir():
            return None
        if (base_dir / "database").is_dir():
            return base_dir / "database", queries_dir
        if (base_dir / "gallery").is_dir():
            return base_dir / "gallery", queries_dir
        return None

    @staticmethod
    def _scan_jpg(folder: Path) -> List[Path]:
        image_paths = sorted(folder.rglob("*.jpg"))
        if not image_paths:
            raise FileNotFoundError(f"No .jpg files found in {folder}")
        return image_paths

    @staticmethod
    def _parse_utm_coords(image_path: Path) -> Tuple[float, float]:
        """
        Parse easting/northing from filename format:
        `...@easting@northing@...`
        """
        match = re.search(r"@(-?\d+(?:\.\d+)?)@(-?\d+(?:\.\d+)?)(?:@|$)", image_path.stem)
        if match is None:
            raise ValueError(
                "Unable to parse UTM coordinates from filename: "
                f"{image_path.name}. Expected pattern ...@easting@northing@..."
            )
        return float(match.group(1)), float(match.group(2))

    def _compute_soft_positives(self) -> List[np.ndarray]:
        """
        Compute radius-based soft positives from query to database using UTM.
        """
        knn = NearestNeighbors(radius=self.positive_dist_threshold, metric="euclidean")
        knn.fit(self.db_coords)
        positives = knn.radius_neighbors(self.q_coords, return_distance=False)
        return [np.asarray(p, dtype=np.int64) for p in positives]

    def _apply_test_method(self, image: Tensor) -> Tensor:
        image = image.to(dtype=torch.float32) / 255.0

        if self.test_method == "hard_resize":
            out = F.resize(image, self.image_size, interpolation=F.InterpolationMode.BICUBIC)
            return self.normalize(out)

        if self.test_method == "central_crop":
            resize_side = max(self.image_size)
            out = F.resize(image, resize_side, interpolation=F.InterpolationMode.BICUBIC)
            out = F.center_crop(out, self.image_size)
            return self.normalize(out)

        # five_crops
        resize_side = max(self.image_size)
        out = F.resize(image, resize_side, interpolation=F.InterpolationMode.BICUBIC)
        crops = F.five_crop(out, self.image_size)
        crops_tensor = torch.stack(list(crops), dim=0)  # [5, C, H, W]
        return self.normalize(crops_tensor)
