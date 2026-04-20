# -*- coding: UTF-8 -*-
"""Training datasets for repo-adapted SAGE.

This module is intentionally kept API-compatible with the public repo's
`datasets_ws.py` wherever possible, while adding the triplet mining utilities
needed by the training script.
"""

import os
import logging
from glob import glob
from os.path import join

import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
from torch.utils.data.dataloader import DataLoader


base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def path_to_pil_img(path):
    return Image.open(path).convert("RGB")



def collate_fn(batch):
    """Collate a batch of triplets.

    Output:
        images: [batch_size*(2+N), 3, H, W]
        triplets_local_indexes: [batch_size*N, 3]
        triplets_global_indexes: [batch_size, 2+N]
    """
    images = torch.cat([e[0] for e in batch])
    triplets_local_indexes = torch.cat([e[1][None] for e in batch])
    triplets_global_indexes = torch.cat([e[2][None] for e in batch])
    for i, (local_indexes, global_indexes) in enumerate(zip(triplets_local_indexes, triplets_global_indexes)):
        local_indexes += len(global_indexes) * i
    return images, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes


class PCADataset(data.Dataset):
    def __init__(self, args, datasets_folder="dataset", dataset_folder="pitts30k/images/train"):
        dataset_folder_full_path = join(datasets_folder, dataset_folder)
        if not os.path.exists(dataset_folder_full_path):
            raise FileNotFoundError(f"Folder {dataset_folder_full_path} does not exist")
        self.images_paths = sorted(glob(join(dataset_folder_full_path, "**", "*.jpg"), recursive=True))
        self.resize = args.resize

    def __getitem__(self, index):
        img = base_transform(path_to_pil_img(self.images_paths[index]))
        img = transforms.functional.resize(img, self.resize)
        return img

    def __len__(self):
        return len(self.images_paths)


class BaseDataset(data.Dataset):
    """Dataset used for inference/testing and cache construction."""

    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train"):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.dataset_folder = join(datasets_folder, dataset_name, "images", split)
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")

        self.resize = args.resize
        self.test_method = args.test_method

        database_folder = join(self.dataset_folder, "database")
        queries_folder = join(self.dataset_folder, "queries")
        if not os.path.exists(database_folder):
            raise FileNotFoundError(f"Folder {database_folder} does not exist")
        if not os.path.exists(queries_folder):
            raise FileNotFoundError(f"Folder {queries_folder} does not exist")

        self.database_paths = sorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))
        self.queries_paths = sorted(glob(join(queries_folder, "**", "*.jpg"), recursive=True))

        # Filename pattern follows the Visual Geo-localization Benchmark:
        # .../@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array([(p.split("@")[1], p.split("@")[2]) for p in self.database_paths]).astype(np.float64)
        self.queries_utms = np.array([(p.split("@")[1], p.split("@")[2]) for p in self.queries_paths]).astype(np.float64)

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.soft_positives_per_query = list(knn.radius_neighbors(
            self.queries_utms,
            radius=args.val_positive_dist_threshold,
            return_distance=False,
        ))

        self.images_paths = list(self.database_paths) + list(self.queries_paths)
        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)
        logging.info(f"[DEBUG] len(queries_paths) = {len(self.queries_paths)}")
        logging.info(f"[DEBUG] len(queries_utms) = {len(self.queries_utms)}")
        logging.info(f"[DEBUG] len(soft_positives_per_query) = {len(self.soft_positives_per_query)}")

    def __getitem__(self, index):
        img = path_to_pil_img(self.images_paths[index])
        img = base_transform(img)
        if self.test_method == "hard_resize":
            img = transforms.functional.resize(img, self.resize)
        else:
            img = self._test_query_transform(img)
        return img, index

    def _test_query_transform(self, img):
        _, h, w = img.shape
        if self.test_method == "single_query":
            processed_img = transforms.functional.resize(img, self.resize)
        elif self.test_method == "central_crop":
            scale = max(self.resize[0] / h, self.resize[1] / w)
            processed_img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale).squeeze(0)
            processed_img = transforms.functional.center_crop(processed_img, self.resize)
            assert processed_img.shape[1:] == torch.Size(self.resize), f"{processed_img.shape[1:]} {self.resize}"
        elif self.test_method in ("five_crops", "nearest_crop", "maj_voting"):
            shorter_side = min(self.resize)
            processed_img = transforms.functional.resize(img, shorter_side)
            processed_img = torch.stack(transforms.functional.five_crop(processed_img, shorter_side))
            assert processed_img.shape == torch.Size([5, 3, shorter_side, shorter_side]), (
                f"{processed_img.shape} {torch.Size([5, 3, shorter_side, shorter_side])}"
            )
        else:
            raise ValueError(f"Unsupported test_method: {self.test_method}")
        return processed_img

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; #queries: {self.queries_num} >"

    def get_positives(self):
        return self.soft_positives_per_query


class RAMEfficient2DMatrix:
    """Sparse-row feature cache used by full-database mining."""

    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.matrix = [None] * shape[0]

    def __setitem__(self, indexes, vals):
        assert vals.shape[1] == self.shape[1], f"{vals.shape[1]} {self.shape[1]}"
        for i, val in zip(indexes, vals):
            self.matrix[i] = val.astype(self.dtype, copy=False)

    def __getitem__(self, index):
        if hasattr(index, "__len__"):
            return np.array([self.matrix[i] for i in index])
        return self.matrix[index]



def _minmax_norm_1d(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


class TripletsDataset(BaseDataset):
    """Training dataset with periodic cache refresh and several mining modes."""

    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train", negs_num_per_query=10):
        super().__init__(args, datasets_folder, dataset_name, split)
        self.mining = args.mining
        self.neg_samples_num = args.neg_samples_num
        self.negs_num_per_query = negs_num_per_query
        self.is_inference = False

        aug = []
        if max(args.brightness, args.contrast, args.saturation, args.hue) > 0:
            aug.append(transforms.ColorJitter(args.brightness, args.contrast, args.saturation, args.hue))
        if args.rand_perspective > 0:
            aug.append(transforms.RandomPerspective(args.rand_perspective))
        if args.random_resized_crop > 0:
            low = max(0.05, 1.0 - float(args.random_resized_crop))
            aug.append(transforms.RandomResizedCrop(size=tuple(self.resize), scale=(low, 1.0)))
        if args.random_rotation > 0:
            aug.append(transforms.RandomRotation(degrees=args.random_rotation))

        self.resized_transform = transforms.Compose([
            transforms.Resize(self.resize) if self.resize is not None else transforms.Lambda(lambda x: x),
            base_transform,
        ])
        self.query_transform = transforms.Compose(aug + [self.resized_transform])

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.hard_positives_per_query = list(
            knn.radius_neighbors(
                self.queries_utms,
                radius=args.train_positives_dist_threshold,
                return_distance=False,
            )
        )

        queries_without_any_hard_positive = np.where(
            np.array([len(p) for p in self.hard_positives_per_query]) == 0
        )[0]

        if len(queries_without_any_hard_positive) != 0:
            logging.info(
                f"There are {len(queries_without_any_hard_positive)} queries without positives "
                "within the training threshold; they are removed."
            )
            keep_mask = np.ones(len(self.hard_positives_per_query), dtype=bool)
            keep_mask[queries_without_any_hard_positive] = False

            self.hard_positives_per_query = [
                self.hard_positives_per_query[i]
                for i in range(len(self.hard_positives_per_query))
                if keep_mask[i]
            ]
            self.soft_positives_per_query = [
                self.soft_positives_per_query[i]
                for i in range(len(self.soft_positives_per_query))
                if keep_mask[i]
            ]
            self.queries_paths = np.array(self.queries_paths)[keep_mask]
            self.queries_utms = self.queries_utms[keep_mask]

        self.queries_paths = list(self.queries_paths)
        self.images_paths = list(self.database_paths) + list(self.queries_paths)
        self.queries_num = len(self.queries_paths)

        logging.info(f"[DEBUG] len(queries_paths) = {len(self.queries_paths)}")
        logging.info(f"[DEBUG] len(queries_utms) = {len(self.queries_utms)}")
        logging.info(f"[DEBUG] len(soft_positives_per_query) = {len(self.soft_positives_per_query)}")
        logging.info(f"[DEBUG] len(hard_positives_per_query) = {len(self.hard_positives_per_query)}")

        assert len(self.queries_paths) == len(self.queries_utms), (
            f"len(queries_paths)={len(self.queries_paths)} vs len(queries_utms)={len(self.queries_utms)}"
        )
        assert len(self.queries_paths) == len(self.soft_positives_per_query), (
            f"len(queries_paths)={len(self.queries_paths)} vs len(soft_positives_per_query)={len(self.soft_positives_per_query)}"
        )
        assert len(self.queries_paths) == len(self.hard_positives_per_query), (
            f"len(queries_paths)={len(self.queries_paths)} vs len(hard_positives_per_query)={len(self.hard_positives_per_query)}"
        )

        if self.mining == "full":
            self.neg_cache = [np.empty((0,), dtype=np.int32) for _ in range(self.queries_num)]

        if self.mining == "msls_weighted":
            notes = [p.split("@")[-2] for p in self.queries_paths]
            try:
                night_indexes = np.where(np.array([n.split("_")[0] == "night" for n in notes]))[0]
                sideways_indexes = np.where(np.array([n.split("_")[1] == "sideways" for n in notes]))[0]
            except IndexError as e:
                raise RuntimeError(
                    "msls_weighted requires Mapillary SLS-style filenames (night/sideways in path)."
                ) from e
            self.weights = np.ones(self.queries_num)
            assert len(night_indexes) != 0 and len(sideways_indexes) != 0, "msls_weighted: missing night/sideways."
            self.weights[night_indexes] += self.queries_num / len(night_indexes)
            self.weights[sideways_indexes] += self.queries_num / len(sideways_indexes)
            self.weights /= self.weights.sum()
            logging.info(
                f"msls_weighted: sideways {len(sideways_indexes)}/{self.queries_num}, "
                f"night {len(night_indexes)}/{self.queries_num}"
            )

    def __getitem__(self, index):
        if self.is_inference:
            return super().__getitem__(index)
        row = self.triplets_global_indexes[index]
        query_index = int(row[0].item())
        best_positive_index = int(row[1].item())
        neg_indexes = row[2:].cpu().numpy().astype(np.int64)
        query = self.query_transform(path_to_pil_img(self.queries_paths[query_index]))
        positive = self.resized_transform(path_to_pil_img(self.database_paths[best_positive_index]))
        negatives = [self.resized_transform(path_to_pil_img(self.database_paths[i])) for i in neg_indexes]
        images = torch.stack((query, positive, *negatives), 0)
        triplets_local_indexes = torch.cat(
            [torch.tensor([[0, 1, 2 + neg_num]], dtype=torch.long) for neg_num in range(len(neg_indexes))],
            dim=0,
        )
        return images, triplets_local_indexes, row

    def __len__(self):
        if self.is_inference:
            return super().__len__()
        return len(self.triplets_global_indexes)

    def compute_triplets(self, args, model):
        self.is_inference = True
        if self.mining == "full":
            self.compute_triplets_full(args, model)
        elif self.mining in ("partial", "msls_weighted"):
            self.compute_triplets_partial(args, model)
        elif self.mining == "random":
            self.compute_triplets_random(args, model)
        elif self.mining == "sage":
            self.compute_triplets_sage(args, model)
        else:
            raise ValueError(f"Unknown mining: {self.mining}")

    @staticmethod
    def compute_cache(args, model, subset_ds, cache_shape):
        subset_dl = DataLoader(
            dataset=subset_ds,
            num_workers=args.num_workers,
            batch_size=args.infer_batch_size,
            shuffle=False,
            pin_memory=(args.device == "cuda"),
        )
        model = model.eval()
        cache = RAMEfficient2DMatrix(cache_shape, dtype=np.float32)
        with torch.no_grad():
            for images, indexes in tqdm(subset_dl, ncols=100, desc="cache"):
                images = images.to(args.device)
                features = model(images)
                cache[indexes.numpy()] = features.cpu().numpy()
        return cache

    def get_query_features(self, query_index, cache):
        qf = cache[query_index + self.database_num]
        if qf is None:
            raise RuntimeError(f"Missing cache for query index {query_index}")
        return qf.reshape(1, -1)

    def get_best_positive_index(self, args, query_index, cache, query_features):
        positives_features = cache[self.hard_positives_per_query[query_index]]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(positives_features)
        _, best_positive_num = faiss_index.search(query_features.reshape(1, -1).astype(np.float32), 1)
        return int(self.hard_positives_per_query[query_index][best_positive_num[0, 0]])

    def get_hardest_negatives_indexes(self, args, cache, query_features, neg_samples):
        neg_features = cache[neg_samples]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(neg_features.astype(np.float32))
        _, neg_nums = faiss_index.search(query_features.reshape(1, -1).astype(np.float32), self.negs_num_per_query)
        neg_nums = neg_nums.reshape(-1)
        return neg_samples[neg_nums].astype(np.int32)

    def _sage_select_negatives(self, args, cache, query_index, query_features, neg_cand):
        if len(neg_cand) == 0:
            return np.random.choice(self.database_num, self.negs_num_per_query, replace=False).astype(np.int32)

        feats = cache[neg_cand].astype(np.float32)
        q = query_features.reshape(-1).astype(np.float32)
        qn = q / (np.linalg.norm(q) + 1e-8)
        fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        vis = (fn @ qn).astype(np.float32)

        q_utm = self.queries_utms[query_index].astype(np.float64)
        db_utm = self.database_utms[neg_cand].astype(np.float64)
        geo_dist = np.linalg.norm(db_utm - q_utm[None, :], axis=1).astype(np.float32)
        geo = 1.0 / (geo_dist + 1.0)

        vis_n = _minmax_norm_1d(vis)
        geo_n = _minmax_norm_1d(geo)
        fused = args.sage_vis_weight * vis_n + args.sage_geo_weight * geo_n

        pool_size = min(int(args.sage_pool_size), len(neg_cand))
        top_local = np.argsort(-fused)[:pool_size]
        pool_idx = neg_cand[top_local]
        pool_feats = cache[pool_idx].astype(np.float32)
        pool_fn = pool_feats / (np.linalg.norm(pool_feats, axis=1, keepdims=True) + 1e-8)

        seed = int(np.argmax(fused[top_local]))
        selected = [seed]
        while len(selected) < self.negs_num_per_query:
            best_j, best_score = None, -1e30
            for j in range(len(pool_idx)):
                if j in selected:
                    continue
                cohesion = sum(float(pool_fn[j] @ pool_fn[i]) for i in selected)
                if cohesion > best_score:
                    best_score = cohesion
                    best_j = j
            if best_j is None:
                break
            selected.append(best_j)

        out = pool_idx[selected].astype(np.int32)
        if len(out) < self.negs_num_per_query:
            extra = np.setdiff1d(neg_cand, out, assume_unique=False)
            if len(extra) == 0:
                extra = np.arange(self.database_num, dtype=np.int32)
            replace = len(extra) < (self.negs_num_per_query - len(out))
            fill = np.random.choice(extra, self.negs_num_per_query - len(out), replace=replace)
            out = np.concatenate([out, fill.astype(np.int32)])
        return out[: self.negs_num_per_query]

    def _sample_valid_queries(self, num, use_weights=False):
        valid_queries_num = len(self.hard_positives_per_query)
        sample_size = min(num, valid_queries_num)
        if use_weights:
            return np.random.choice(valid_queries_num, sample_size, replace=False, p=self.weights)
        return np.random.choice(valid_queries_num, sample_size, replace=False)

    def compute_triplets_random(self, args, model):
        self.triplets_global_indexes = []
        sampled_queries_indexes = self._sample_valid_queries(args.cache_refresh_rate)
        positives_indexes = [self.hard_positives_per_query[i] for i in sampled_queries_indexes]
        positives_indexes = [p for pos in positives_indexes for p in pos]
        positives_indexes = list(np.unique(positives_indexes))
        subset_ds = Subset(self, positives_indexes + list(sampled_queries_indexes + self.database_num))
        cache = self.compute_cache(args, model, subset_ds, (len(self), args.features_dim))
        for query_index in tqdm(sampled_queries_indexes, ncols=100, desc="triplets-random"):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.random.choice(
                self.database_num,
                size=self.negs_num_per_query + len(soft_positives),
                replace=False,
            )
            neg_indexes = np.setdiff1d(neg_indexes, soft_positives, assume_unique=True)[: self.negs_num_per_query]
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes, dtype=torch.long)

    def compute_triplets_full(self, args, model):
        self.triplets_global_indexes = []
        sampled_queries_indexes = self._sample_valid_queries(args.cache_refresh_rate)
        database_indexes = list(range(self.database_num))
        subset_ds = Subset(self, database_indexes + list(sampled_queries_indexes + self.database_num))
        cache = self.compute_cache(args, model, subset_ds, (len(self), args.features_dim))
        for query_index in tqdm(sampled_queries_indexes, ncols=100, desc="triplets-full"):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            neg_indexes = np.random.choice(self.database_num, self.neg_samples_num, replace=False)
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.setdiff1d(neg_indexes, soft_positives, assume_unique=True)
            neg_indexes = np.unique(np.concatenate([self.neg_cache[query_index], neg_indexes]))
            neg_indexes = self.get_hardest_negatives_indexes(args, cache, query_features, neg_indexes)
            self.neg_cache[query_index] = neg_indexes
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes, dtype=torch.long)

    def compute_triplets_partial(self, args, model):
        self.triplets_global_indexes = []
        if self.mining == "partial":
            sampled_queries_indexes = self._sample_valid_queries(args.cache_refresh_rate)
        else:
            sampled_queries_indexes = self._sample_valid_queries(args.cache_refresh_rate, use_weights=True)
        sampled_database_indexes = np.random.choice(self.database_num, self.neg_samples_num, replace=False)
        positives_indexes = [self.hard_positives_per_query[i] for i in sampled_queries_indexes]
        positives_indexes = [p for pos in positives_indexes for p in pos]
        database_indexes = list(np.unique(list(sampled_database_indexes) + positives_indexes))
        subset_ds = Subset(self, database_indexes + list(sampled_queries_indexes + self.database_num))
        cache = self.compute_cache(args, model, subset_ds, (len(self), args.features_dim))
        for query_index in tqdm(sampled_queries_indexes, ncols=100, desc="triplets-partial"):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.setdiff1d(sampled_database_indexes, soft_positives, assume_unique=True)
            neg_indexes = self.get_hardest_negatives_indexes(args, cache, query_features, neg_indexes)
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes, dtype=torch.long)

    def compute_triplets_sage(self, args, model):
        self.triplets_global_indexes = []
        sampled_queries_indexes = self._sample_valid_queries(args.cache_refresh_rate)
        sampled_database_indexes = np.random.choice(self.database_num, self.neg_samples_num, replace=False)
        positives_indexes = [self.hard_positives_per_query[i] for i in sampled_queries_indexes]
        positives_indexes = [p for pos in positives_indexes for p in pos]
        database_indexes = list(np.unique(list(sampled_database_indexes) + positives_indexes))
        subset_ds = Subset(self, database_indexes + list(sampled_queries_indexes + self.database_num))
        cache = self.compute_cache(args, model, subset_ds, (len(self), args.features_dim))
        for query_index in tqdm(sampled_queries_indexes, ncols=100, desc="triplets-sage"):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            soft_positives = self.soft_positives_per_query[query_index]
            neg_cand = np.setdiff1d(sampled_database_indexes, soft_positives, assume_unique=True)
            neg_indexes = self._sage_select_negatives(args, cache, query_index, query_features, neg_cand)
            self.triplets_global_indexes.append((query_index, best_positive_index, *tuple(neg_indexes)))
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes, dtype=torch.long)
