# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
import lightning as L
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2 as T

from src.dataloaders import GSVCitiesDataset
from src.dataloaders import PittsburghDataset
from src.dataloaders import MapillarySLSDataset
from src.dataloaders import PairedNameTestDataset, NordlandSequenceDataset, CoordinateRadiusTestDataset


TEST_DATASET_BUILDERS = {
    # 这里统一登记“测试集名字 -> 对应 loader 构造逻辑”。
    # 不同 benchmark 的 ground truth 定义不一样，所以不能都用同一种 dataset 类。
    "pitts30k-test": lambda path, transform: CoordinateRadiusTestDataset(
        dataset_path=path,
        dataset_name="pitts30k-test",
        radius_m=25.0,
        reference_subdir="test/database",
        query_subdirs=["test/queries"],
        transform=transform,
    ),
    "nordland": lambda path, transform: NordlandSequenceDataset(
        dataset_path=path,
        dataset_name="nordland",
        reference_subdir="test/database",
        query_subdir="test/queries",
        frame_tolerance=10,
        transform=transform,
    ),
    "sped": lambda path, transform: PairedNameTestDataset(
        dataset_path=path,
        dataset_name="sped",
        reference_subdir="test/database",
        query_subdir="test/queries",
        transform=transform,
    ),
    "amstertime": lambda path, transform: PairedNameTestDataset(
        dataset_path=path,
        dataset_name="amstertime",
        reference_subdir="test/database",
        query_subdir="test/queries",
        transform=transform,
    ),
    "tokyo247": lambda path, transform: CoordinateRadiusTestDataset(
        dataset_path=path,
        dataset_name="tokyo247",
        radius_m=25.0,
        reference_subdir="test/database",
        query_subdirs=["test/queries"],
        transform=transform,
    ),
    "svox-all": lambda path, transform: CoordinateRadiusTestDataset(
        dataset_path=path,
        dataset_name="svox-all",
        radius_m=25.0,
        reference_subdir="test/gallery",
        query_subdirs=[
            "test/queries",
            "test/queries_sun",
            "test/queries_snow",
            "test/queries_rain",
            "test/queries_night",
            "test/queries_overcast",
        ],
        transform=transform,
        drop_queries_without_positives=True,
    ),
}


class VPRDataModule(L.LightningDataModule):
    # 这个类是 Lightning 管数据的标准入口。
    # train.py 不直接关心数据细节，只和它打交道。
    def __init__(
        self,
        gsv_cities_path: str = None,
        cities: str | list = "all",
        img_per_place: int = 4,
        val_sets: dict = {"msls-val": None, "pitts30k-val": None},
        test_sets: dict | None = None,
        periodic_test: bool = False,
        train_img_size=(224, 224),
        val_img_size=(224, 224),
        batch_size: int = 100,
        eval_batch_size: int | None = None,
        num_workers: int = 8,
        shuffle: bool = False,
        mean_std: dict = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    ):
        super().__init__()
        self.gsv_cities_path = gsv_cities_path
        self.cities = cities
        self.img_per_place = img_per_place
        self.val_sets = val_sets
        self.test_sets = test_sets or {}
        self.periodic_test = periodic_test
        self.train_img_size = train_img_size
        self.val_img_size = val_img_size
        self.mean_std = mean_std
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        # 下面这几个成员是为了让 model.py 能区分:
        # “前几个 loader 是 val，后几个 loader 是 periodic test”。
        self.num_val_datasets = 0
        self.num_test_datasets = 0
        self.val_datasets = []
        self.test_datasets = []

        # 训练时用数据增强，提升鲁棒性。
        self.train_transform = T.Compose([
            T.Resize(train_img_size, interpolation=3),
            T.RandAugment(num_ops=3, magnitude=15, interpolation=2),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(**mean_std),
        ])

        # 验证/测试时不用随机增强，保证结果可复现、可对比。
        self.val_transform = T.Compose([
            T.Resize(val_img_size, interpolation=3),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(**mean_std),
        ])

    # 统一构建 test_datasets 列表，供 test_dataloader 或 periodic_test 复用。
    def _build_test_datasets(self):
        self.test_datasets = []
        for dataset_name, dataset_path in self.test_sets.items():
            if dataset_name not in TEST_DATASET_BUILDERS:
                raise ValueError(f"Unsupported test dataset: {dataset_name}")
            self.test_datasets.append(TEST_DATASET_BUILDERS[dataset_name](dataset_path, self.val_transform))
        self.num_test_datasets = len(self.test_datasets)

    def setup(self, stage=None):
        # Lightning 会在 fit / test 等不同阶段调用 setup。
        if stage in ["fit", "reload", None]:
            # fit/reload: 构建训练集。
            self.train_dataset = GSVCitiesDataset(
                dataset_path=self.gsv_cities_path,
                cities=self.cities,
                img_per_place=self.img_per_place,
                transform=self.train_transform,
            )

        if stage in ["fit", None]:
            # fit 阶段除了训练集，还要准备验证集。
            self.val_datasets = []
            if "msls-val" in self.val_sets:
                self.val_datasets.append(
                    MapillarySLSDataset(
                        dataset_path=self.val_sets["msls-val"],
                        transform=self.val_transform,
                    )
                )
            if "pitts30k-val" in self.val_sets:
                self.val_datasets.append(
                    PittsburghDataset(
                        dataset_path=self.val_sets["pitts30k-val"],
                        transform=self.val_transform,
                    )
                )
            self.num_val_datasets = len(self.val_datasets)

            # 如果打开 periodic_test，就在 fit 阶段顺便把 test datasets 也挂进来。
            if self.periodic_test and self.test_sets:
                self._build_test_datasets()

        if stage in ["test", None]:
            # 独立 test 模式下，也要单独构建测试集。
            self._build_test_datasets()

    def train_dataloader(self):
        # 每个 epoch 前刷新训练集，让 GSV-Cities 的采样随 epoch 更新。
        self.setup(stage="reload")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        # 默认只返回验证集 loader。
        # 但如果 periodic_test=True，这里会返回:
        # [全部验证集] + [全部测试集]
        eval_datasets = list(self.val_datasets)
        if self.periodic_test:
            eval_datasets.extend(self.test_datasets)
        return [
            DataLoader(
                eval_ds,
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
            )
            for eval_ds in eval_datasets
        ]

    def test_dataloader(self):
        # 独立 test 模式仍然保留标准接口: 一个测试集对应一个 dataloader。
        return [
            DataLoader(
                test_ds,
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                # 批量循环测试多个 checkpoint 时，每轮 test 都会重新创建迭代器。
                # 不保留 worker，避免多轮测试后文件句柄/pipe 释放不及时。
                persistent_workers=False,
            )
            for test_ds in self.test_datasets
        ]
