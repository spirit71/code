import os
import torch
import faiss
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from os.path import join
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def path_to_pil_img(path):
    return Image.open(path).convert("RGB")


def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (images,
        triplets_local_indexes, triplets_global_indexes).
        triplets_local_indexes are the indexes referring to each triplet within images.
        triplets_global_indexes are the global indexes of each image.
    Args:
        batch: list of tuple (images, triplets_local_indexes, triplets_global_indexes).
            considering each query to have 10 negatives (negs_num_per_query=10):
            - images: torch tensor of shape (12, 3, h, w).
            - triplets_local_indexes: torch tensor of shape (10, 3).
            - triplets_global_indexes: torch tensor of shape (12).
    Returns:
        images: torch tensor of shape (batch_size*12, 3, h, w).
        triplets_local_indexes: torch tensor of shape (batch_size*10, 3).
        triplets_global_indexes: torch tensor of shape (batch_size, 12).
    """
    images = torch.cat([e[0] for e in batch])
    triplets_local_indexes = torch.cat([e[1][None] for e in batch])
    triplets_global_indexes = torch.cat([e[2][None] for e in batch])
    for i, (local_indexes, global_indexes) in enumerate(zip(triplets_local_indexes, triplets_global_indexes)):
        local_indexes += len(global_indexes) * i  # Increment local indexes by offset (len(global_indexes) is 12)
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
    """Dataset with images from database and queries, used for inference (testing and building cache).
    """

    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train"):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.dataset_folder = join(datasets_folder, dataset_name, split)
        if not os.path.exists(self.dataset_folder): raise FileNotFoundError(
            f"Folder {self.dataset_folder} does not exist")

        self.resize = args.resize
        self.test_method = args.test_method

        # Read paths and UTM coordinates for database images
        possible_db_folders = ["database", "gallery", "db"]
        database_folder = None
        for folder in possible_db_folders:
            if os.path.exists(join(self.dataset_folder, folder)):
                database_folder = join(self.dataset_folder, folder)
                break
        if database_folder is None:
            raise FileNotFoundError(f"No valid database folder found in {self.dataset_folder}")
        
        self.database_paths = sorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))
        self.database_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(np.float64)

        # Handle multiple query folders for SVOX dataset
        self.queries_folders = self._get_queries_folders()
        self.queries_data = {}  # Store queries data for each folder
        
        # Process each query folder
        for query_folder in self.queries_folders:
            queries_folder_path = join(self.dataset_folder, query_folder)
            if not os.path.exists(queries_folder_path):
                print(f"Warning: Query folder {queries_folder_path} does not exist, skipping")
                continue
                
            queries_paths = sorted(glob(join(queries_folder_path, "**", "*.jpg"), recursive=True))
            if len(queries_paths) == 0:
                print(f"Warning: No images found in {queries_folder_path}, skipping")
                continue
                
            queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in queries_paths]).astype(np.float64)
            
            # Find soft positives for this query set
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.database_utms)
            soft_positives_per_query = knn.radius_neighbors(queries_utms,
                                                          radius=args.val_positive_dist_threshold,
                                                          return_distance=False)
            
            self.queries_data[query_folder] = {
                'paths': queries_paths,
                'utms': queries_utms,
                'soft_positives': soft_positives_per_query,
                'num': len(queries_paths)
            }
        
        if len(self.queries_data) == 0:
            raise FileNotFoundError(f"No valid query folders found in {self.dataset_folder}")

        # Combine all images paths (database + all queries)
        self.images_paths = list(self.database_paths)
        self.query_folder_indices = []  # Track which query folder each image belongs to
        
        for folder_name, data in self.queries_data.items():
            self.images_paths.extend(data['paths'])
            # Store indices for each query folder (folder_name, start_index, end_index)
            start_idx = len(self.images_paths) - len(data['paths'])
            end_idx = len(self.images_paths)
            self.query_folder_indices.append((folder_name, start_idx, end_idx))

        self.database_num = len(self.database_paths)
        self.total_queries_num = sum([data['num'] for data in self.queries_data.values()])

    def _get_queries_folders(self):
        """Get list of query folders based on dataset type."""
        if "svox" in self.dataset_name.lower():
            # SVOX dataset has multiple query conditions
            return ["queries", "queries_night", "queries_overcast", 
                   "queries_rain", "queries_snow", "queries_sun"]
        else:
            # Default dataset only has 'queries' folder
            return ["queries"]

    def __getitem__(self, index):
        img = path_to_pil_img(self.images_paths[index])
        img = base_transform(img)
        if index >= len(self.images_paths):
            raise IndexError(f"Index {index} out of range (max: {len(self.images_paths)-1})")
        # Determine if this is a database or query image
        if index < self.database_num:
            # Database image - use hard_resize
            if self.test_method == "hard_resize":
                img = transforms.functional.resize(img, self.resize)
        else:
            # Query image - apply appropriate transform
            if self.test_method == "hard_resize":
                img = transforms.functional.resize(img, self.resize)
            else:
                img = self._test_query_transform(img)
        return img, index

    def _test_query_transform(self, img):
        """Transform query image according to self.test_method."""
        C, H, W = img.shape
        if self.test_method == "single_query":
            processed_img = transforms.functional.resize(img, self.resize)
        elif self.test_method == "central_crop":
            scale = max(self.resize[0] / H, self.resize[1] / W)
            processed_img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale).squeeze(0)
            processed_img = transforms.functional.center_crop(processed_img, self.resize)
            assert processed_img.shape[1:] == torch.Size(self.resize), f"{processed_img.shape[1:]} {self.resize}"
        elif self.test_method == "five_crops" or self.test_method == 'nearest_crop' or self.test_method == 'maj_voting':
            shorter_side = min(self.resize)
            processed_img = transforms.functional.resize(img, shorter_side)
            processed_img = torch.stack(transforms.functional.five_crop(processed_img, shorter_side))
            assert processed_img.shape == torch.Size([5, 3, shorter_side, shorter_side]), \
                f"{processed_img.shape} {torch.Size([5, 3, shorter_side, shorter_side])}"
        return processed_img

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        query_info = ", ".join([f"{folder}: {data['num']}" for folder, data in self.queries_data.items()])
        return (f"< {self.__class__.__name__}, {self.dataset_name} - "
                f"#database: {self.database_num}; #queries: {query_info} >")

    def get_positives(self):
        """Return soft positives for all query folders."""
        return self.queries_data

    def get_query_folder_info(self):
        """Return information about query folder indices."""
        return self.query_folder_indices

    def get_database_paths(self):
        """Return database image paths."""
        return self.database_paths

    def get_queries_paths_by_folder(self, folder_name):
        """Return query paths for specific folder."""
        if folder_name in self.queries_data:
            return self.queries_data[folder_name]['paths']
        return []
    def get_global_index(self, folder_name, local_index):
    # """将查询文件夹内的局部索引转换为全局索引"""
        for name, start, end in self.query_folder_indices:
            if name == folder_name:
                return start + local_index
        raise ValueError(f"Folder {folder_name} not found")