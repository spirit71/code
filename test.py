import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

def test(args, eval_ds, model, test_method="hard_resize", pca=None):
    """Compute features of the given dataset and compute the recalls."""
    assert test_method in ["hard_resize", "single_query", "central_crop", "five_crops",
                          "nearest_crop", "maj_voting"], f"test_method can't be {test_method}"

    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        
        # Initialize features array
        if test_method == "nearest_crop" or test_method == 'maj_voting':
            # For multi-crop methods, we need space for 5 crops per query
            total_queries = sum([data['num'] for data in eval_ds.queries_data.values()])
            all_features = np.empty((5 * total_queries + eval_ds.database_num, args.features_dim), dtype="float32")
        else:
            all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")

        # Extract database features
        for inputs, indices in tqdm(database_dataloader, ncols=100):
            features = model(inputs.to(args.device))
            features = features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features
        
        # Extract queries features for each query folder
        queries_features_dict = {}
        queries_indices_dict = {}
        
        for folder_name, data in eval_ds.queries_data.items():
            logging.debug(f"Extracting features for query folder: {folder_name}")
            
             # 获取当前查询文件夹在images_paths中的实际位置
            folder_paths = data['paths']
            # global_indices = [eval_ds.images_paths.index(p) for p in folder_paths]
            start_idx = eval_ds.images_paths.index(folder_paths[0])  # 第一个图像的全局索引
            end_idx = start_idx + len(folder_paths)  # 最后一个图像的全局索引+1
            
            # 确保索引不越界
            if end_idx > len(eval_ds.images_paths):
                raise ValueError(f"Index {end_idx} out of range for {folder_name}")
            
            queries_subset_ds = Subset(eval_ds, list(range(start_idx, end_idx)))
            queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
            queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                          batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
            
            # Initialize array for this query set's features
            if test_method == "nearest_crop" or test_method == 'maj_voting':
                folder_features = np.empty((5 * data['num'], args.features_dim), dtype="float32")
            else:
                folder_features = np.empty((data['num'], args.features_dim), dtype="float32")
            
            # Process each batch
            for inputs, indices in tqdm(queries_dataloader, ncols=100):
                if test_method == "five_crops" or test_method == "nearest_crop" or test_method == 'maj_voting':
                    inputs = torch.cat(tuple(inputs))  # shape = 5*bs x 3 x 480 x 480
                
                features = model(inputs.to(args.device))
                
                if test_method == "five_crops":  # Compute mean along the 5 crops
                    features = torch.stack(torch.split(features, 5)).mean(1)
                
                features = features.cpu().numpy()
                if pca != None:
                    features = pca.transform(features)
                
                if test_method == "nearest_crop" or test_method == 'maj_voting':
                    # For multi-crop methods, store all 5 crops
                    batch_start_idx = (indices[0] - start_idx) * 5
                    batch_end_idx = batch_start_idx + indices.shape[0] * 5
                    batch_indices = np.arange(batch_start_idx, batch_end_idx)
                    folder_features[batch_indices, :] = features
                else:
                    # For single image methods, store directly
                    batch_indices = indices.numpy() - start_idx
                    folder_features[batch_indices, :] = features
            
            queries_features_dict[folder_name] = folder_features
            queries_indices_dict[folder_name] = (start_idx, end_idx)
            
            # Store in all_features array
            if test_method == "nearest_crop" or test_method == 'maj_voting':
                all_features[start_idx*5 : end_idx*5] = folder_features
            else:
                all_features[start_idx:end_idx] = folder_features
    
    # Compute recalls for each query folder
    all_recalls = {}
    database_features = all_features[:eval_ds.database_num]
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del database_features, all_features
    
    for folder_name, data in eval_ds.queries_data.items():
        logging.debug(f"Calculating recalls for {folder_name}")
        queries_features = queries_features_dict[folder_name]
        
        if test_method == "five_crops":
            # Average features across 5 crops
            queries_features = np.stack([np.mean(queries_features[i:i+5], axis=0) 
                                       for i in range(0, len(queries_features), 5)])
        elif test_method == "nearest_crop" or test_method == 'maj_voting':
            # Use voting method
            queries_features = queries_features.reshape(-1, 5, args.features_dim)
            distances, predictions = [], []
            for i in range(5):
                dist, pred = faiss_index.search(queries_features[:, i, :], max(args.recall_values))
                distances.append(dist)
                predictions.append(pred)
            distances = np.stack(distances, axis=1)
            predictions = np.stack(predictions, axis=1)
            
            if test_method == 'maj_voting':
                distances = top_n_voting('top5', predictions, distances, args.maj_weight)
                # After voting, get final predictions
                _, predictions = faiss_index.search(queries_features.mean(axis=1), max(args.recall_values))
            else:
                # For nearest crop, choose the crop with smallest distance
                best_crop = np.argmin(distances[:, :, 0], axis=1)
                predictions = predictions[np.arange(len(predictions)), best_crop]
        else:
            # Standard search
            distances, predictions = faiss_index.search(queries_features, max(args.recall_values))
        
        # Calculate recalls
        positives_per_query = data['soft_positives']
        recalls = np.zeros(len(args.recall_values))
        for query_index, pred in enumerate(predictions):
            for i, n in enumerate(args.recall_values):
                if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break
        
        recalls = recalls / data['num'] * 100
        recalls_str = ", ".join([f"R@{val}: {rec:.3f}" for val, rec in zip(args.recall_values, recalls)])
        all_recalls[folder_name] = (recalls, recalls_str)
    
    # Combine all results
    combined_recalls = np.mean([rec[0] for rec in all_recalls.values()], axis=0)
    combined_str = ", ".join([f"R@{val}: {rec:.3f}" for val, rec in zip(args.recall_values, combined_recalls)])
    
    # Log individual and combined results
    for folder_name, (_, recalls_str) in all_recalls.items():
        logging.info(f"Recalls for {folder_name}: {recalls_str}")
    logging.info(f"Combined recalls: {combined_str}")
    
    return all_recalls, combined_str

def top_n_voting(topn, predictions, distances, maj_weight):
    """Apply majority voting to refine distances."""
    if topn == 'top1':
        n = 1
        selected = 0
    elif topn == 'top5':
        n = 5
        selected = slice(0, 5)
    elif topn == 'top10':
        n = 10
        selected = slice(0, 10)
    
    # Find predictions that repeat in the selected top-n for each crop
    vals, counts = np.unique(predictions[:, :, selected], return_counts=True)
    
    # For each prediction that repeats more than once, adjust its score
    for val, count in zip(vals[counts > 1], counts[counts > 1]):
        mask = (predictions[:, :, selected] == val)
        distances[:, :, selected][mask] -= maj_weight * count/n
    
    return distances