# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
import lightning as L
import warnings
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2  as T

from src.dataloaders import GSVCitiesDataset
from src.dataloaders import PittsburghDataset
from src.dataloaders import MapillarySLSDataset
from src.dataloaders import GenericVPRTestDataset

class VPRDataModule(L.LightningDataModule):
    def __init__(
        self,
        gsv_cities_path: str = None,
        cities: str | list = "all",
        img_per_place: int = 4,
        val_sets: dict | None = None,
        test_sets: dict | None = None,
        train_img_size=(224, 224),
        val_img_size=(224, 224),
        batch_size: int = 100,
        num_workers: int = 8,
        shuffle: bool = False,
        mean_std: dict = {"mean":[0.485, 0.456, 0.406], "std":[0.229, 0.224, 0.225]},
    ):
        super().__init__()
        self.gsv_cities_path = gsv_cities_path
        self.cities = cities
        self.img_per_place = img_per_place
        self.val_sets = val_sets or {}
        self.test_sets = test_sets or {}
        self.train_img_size = train_img_size
        self.val_img_size = val_img_size
        self.mean_std = mean_std
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        
        self.train_transform = T.Compose([
            T.Resize(train_img_size, interpolation=3),
            T.RandAugment(num_ops=3, magnitude=15, interpolation=2),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(**mean_std),
        ])
        
        self.val_transform = T.Compose([
            T.Resize(val_img_size, interpolation=3),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(**mean_std),
        ])

    def _build_eval_datasets(self, datasets_cfg: dict, split_name: str):
        dataset_factories = {
            "msls-val": MapillarySLSDataset,
            "pitts30k-val": PittsburghDataset,
        }
        generic_test_set_names = {
            "nordland",
            "sped",
            "amstertime",
            "tokyo247",
            "svox",
            "pitts30k-test",
        }
        supported_names = list(dataset_factories.keys()) + sorted(generic_test_set_names)

        datasets = []
        unsupported = []

        for dataset_name, dataset_cfg in datasets_cfg.items():
            dataset_path = dataset_cfg
            split = "test"
            test_method = "hard_resize"
            positive_dist_threshold = 10.0 if dataset_name == "nordland" else 25.0

            if isinstance(dataset_cfg, dict):
                dataset_path = dataset_cfg.get("path")
                split = dataset_cfg.get("split", split)
                test_method = dataset_cfg.get("test_method", test_method)
                positive_dist_threshold = dataset_cfg.get(
                    "positive_dist_threshold", positive_dist_threshold
                )

            if dataset_path is None:
                unsupported.append(dataset_name)
                continue

            dataset_cls = dataset_factories.get(dataset_name)
            try:
                if dataset_cls is not None:
                    datasets.append(
                        dataset_cls(
                            dataset_path=dataset_path,
                            transform=self.val_transform,
                        )
                    )
                    continue

                if split_name == "test" and dataset_name in generic_test_set_names:
                    datasets.append(
                        GenericVPRTestDataset(
                            dataset_path=dataset_path,
                            split=split,
                            test_method=test_method,
                            image_size=self.val_img_size,
                            positive_dist_threshold=positive_dist_threshold,
                            dataset_name=dataset_name,
                            mean=self.mean_std["mean"],
                            std=self.mean_std["std"],
                        )
                    )
                    continue
            except Exception as exc:
                warnings.warn(
                    f"Skipping dataset '{dataset_name}' due to setup error: {exc}",
                    stacklevel=2,
                )
                continue

            unsupported.append(dataset_name)

        if unsupported:
            warnings.warn(
                f"Skipping unsupported {split_name} dataset(s): {unsupported}. "
                f"Supported names are: {supported_names}",
                stacklevel=2,
            )

        if datasets_cfg and not datasets:
            raise ValueError(
                f"No valid {split_name} datasets were created. "
                f"Configured keys: {list(datasets_cfg.keys())}. "
                f"Supported keys: {supported_names}."
            )

        return datasets

    def setup(self, stage=None):
        if stage in ["fit", "reload", None]:
            self.train_dataset = GSVCitiesDataset(
                dataset_path=self.gsv_cities_path,
                cities=self.cities,
                img_per_place=self.img_per_place,
                transform=self.train_transform
            )
        
        if stage == "fit" or stage is None:
            self.val_datasets = self._build_eval_datasets(self.val_sets, split_name="validation")

        if stage == "test" or stage is None:
            test_cfg = self.test_sets or self.val_sets
            self.test_datasets = self._build_eval_datasets(test_cfg, split_name="test")
        
    def train_dataloader(self):
        self.setup(stage="reload") # reload the train dataset
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=self.shuffle
        )

    def val_dataloader(self):
        return [
            DataLoader(
                val_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            ) for val_ds in self.val_datasets
        ]

    def test_dataloader(self):
        return [
            DataLoader(
                test_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            ) for test_ds in self.test_datasets
        ]
