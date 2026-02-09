"""
叠加热力图分析和检索错误分析工具
用于分析模型在不同数据集上的表现差异，找出检索错误样本并可视化attention
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from PIL import Image
import torchvision.transforms as transforms
import faiss
from collections import defaultdict
import cv2
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from os.path import join
import sys
sys.path.append('..')
import datasets_ws 
import network
import test
import parser
import commons


def extract_attention_weights(model, images, device, patch_grid_h=16, patch_grid_w=16):
    """
    提取模型的attention权重（所有layer和query）
    
    Args:
        model: 模型
        images: 输入图像 [B, C, H, W]
        device: 设备
        patch_grid_h: patch网格高度
        patch_grid_w: patch网格宽度
    
    Returns:
        attention_weights: List of [num_queries, num_keys] attention weights for each layer
        patch_grid_h, patch_grid_w: 实际的patch网格大小
    """
    model.eval()
    model.module._return_stats = True
    
    with torch.no_grad():
        # 获取backbone输出
        x = model.module.backbone(images.to(device))
        
        B, P, D = x["x_norm"].shape
        queries = model.module.queries.expand(B, -1, -1)
        
        x_c = x["x_norm_clstoken"]
        x_p = x["x_norm_patchtokens"]
        x_cp = torch.cat([x_c, x_p], dim=1)
        x_cp = model.module.fc(x_cp)
        
        # 计算patch坐标
        num_patches = x_p.shape[1]
        patch_grid_h = patch_grid_w = int(num_patches ** 0.5)
        
        coords_h = torch.arange(0.5, patch_grid_h, device=x_p.device, dtype=x_p.dtype) / patch_grid_h
        coords_w = torch.arange(0.5, patch_grid_w, device=x_p.device, dtype=x_p.dtype) / patch_grid_w
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), 
            dim=-1
        )
        patch_coords = coords.flatten(0, 1)
        patch_coords = 2.0 * patch_coords - 1.0
        
        # 获取decoder输出和统计信息
        decoder_result = model.module.decoder(
            queries, x_cp,
            patch_grid_h=patch_grid_h,
            patch_grid_w=patch_grid_w,
            patch_coords=patch_coords,
            return_stats=True,
        )
        
        if isinstance(decoder_result, tuple):
            _, decoder_stats = decoder_result
            attention_weights = []
            for layer_stats in decoder_stats:
                if layer_stats and 'attn_weights_sample' in layer_stats:
                    attention_weights.append(layer_stats['attn_weights_sample'])
            return attention_weights, patch_grid_h, patch_grid_w
    return [], patch_grid_h, patch_grid_w


def overlay_attention_on_image(image: np.ndarray, attention_map: np.ndarray, 
                               alpha: float = 0.6, sigma: float = 5.0):
    """
    将attention热力图叠加到原始图像上
    改进：使用Attention强度作为Mask，低关注区域透明，保留原图原貌。
    
    Args:
        image: 原始图像 [H, W, 3], uint8, 0-255
        attention_map: attention权重图 [H_attn, W_attn], float
        alpha: 热力图的最大不透明度 (0.0 - 1.0)
        sigma: 高斯平滑参数 (建议 3.0 - 5.0，太大会丢失细节)
    
    Returns:
        overlaid_image: 叠加后的图像 [H, W, 3], uint8, RGB格式
    """
    # 1. 基础校验
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    img_h, img_w = image.shape[:2]
    
    # 2. 尺寸对齐：双三次插值放大
    attention_map = cv2.resize(
        attention_map.astype(np.float32),
        (img_w, img_h),
        interpolation=cv2.INTER_CUBIC
    )
    
    # 3. 归一化到 0-1 之间 (用于计算Mask)
    # 截断极端值以增加对比度
    p2 = np.percentile(attention_map, 2)
    p98 = np.percentile(attention_map, 98)
    attention_map = np.clip(attention_map, p2, p98)
    
    attn_min, attn_max = attention_map.min(), attention_map.max()
    if attn_max > attn_min:
        attn_norm = (attention_map - attn_min) / (attn_max - attn_min)
    else:
        attn_norm = np.zeros_like(attention_map)
        
    # 4. 高斯平滑 (适度平滑，保留结构)
    # 在归一化后平滑，效果更自然
    attn_norm = gaussian_filter(attn_norm, sigma=sigma)
    # 再次归一化确保范围正确
    attn_norm = (attn_norm - attn_norm.min()) / (attn_norm.max() - attn_norm.min() + 1e-8)
    
    # 5. 生成热力图颜色 (Jet colormap)
    heatmap_uint8 = (attn_norm * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) # 转回RGB
    
    # 6. 图像融合 (关键改进步骤)
    # 逻辑：Attention值越高，热力图越不透明；Attention值越低，直接显示原图
    
    # 将图像转为float计算
    img_float = image.astype(np.float32) / 255.0
    heatmap_float = heatmap_colored.astype(np.float32) / 255.0
    
    # 扩展attn_norm维度以匹配RGB通道 [H, W] -> [H, W, 1]
    mask = attn_norm[..., np.newaxis]
    
    # 动态混合公式：
    # Output = Original * (1 - mask * alpha) + Heatmap * (mask * alpha)
    # 当 mask -> 0 (蓝色区域): 权重接近 0，显示原图
    # 当 mask -> 1 (红色区域): 权重接近 alpha，显示热力图颜色
    
    blended = img_float * (1.0 - mask * alpha) + heatmap_float * (mask * alpha)
    
    return np.clip(blended * 255, 0, 255).astype(np.uint8)


def visualize_stacked_heatmaps(attention_weights_list: List[torch.Tensor],
                              image: torch.Tensor,
                              save_path: str,
                              patch_grid_h: int = 16,
                              patch_grid_w: int = 16,
                              num_queries_to_show: int = 4):
    """
    可视化叠加热力图
    """
    num_layers = len(attention_weights_list)
    if num_layers == 0:
        return
    
    # 准备原始图像 (转换为numpy HWC)
    img_tensor = image.cpu()
    img_np = img_tensor.permute(1, 2, 0).numpy()
    
    # 确保原图是 uint8 0-255 范围，且不做任何额外的颜色变换
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)
    
    # 如果是单通道灰度图，转为RGB以保持一致性
    if img_np.shape[2] == 1:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    # 计算Attention数据的逻辑保持不变...
    layer_query_attns = []
    aggregated_attn = None
    
    for layer_idx, attn_weights in enumerate(attention_weights_list):
        if attn_weights is None:
            continue
        
        layer_queries = []
        if attn_weights.dim() == 2:
            num_queries, num_keys = attn_weights.shape
            if num_keys > patch_grid_h * patch_grid_w:
                attn_weights = attn_weights[:, -patch_grid_h * patch_grid_w:]
            num_show = min(num_queries_to_show, num_queries)
            for q_idx in range(num_show):
                q_attn = attn_weights[q_idx].numpy().reshape(patch_grid_h, patch_grid_w)
                layer_queries.append(q_attn)
            layer_attn = attn_weights.mean(dim=0).numpy()
        else:
            attn = attn_weights.numpy()
            if len(attn) > patch_grid_h * patch_grid_w:
                attn = attn[-patch_grid_h * patch_grid_w:]
            q_attn = attn.reshape(patch_grid_h, patch_grid_w)
            layer_queries.append(q_attn)
            layer_attn = attn
        
        layer_query_attns.append(layer_queries)
        if aggregated_attn is None:
            aggregated_attn = layer_attn.copy()
        else:
            aggregated_attn += layer_attn
            
    if aggregated_attn is not None:
        aggregated_attn = aggregated_attn / num_layers
        aggregated_attn = aggregated_attn.reshape(patch_grid_h, patch_grid_w)

    # --- 绘图逻辑优化 ---
    
    num_cols = 2 + num_layers
    # 增加 figsize 的高度，防止标题重叠
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 6))
    
    # 1. 最左侧：完全原始的图像 (不加任何滤镜)
    axes[0].imshow(img_np) 
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    # 2. 中间列：Layer Heatmaps
    for layer_idx, query_attns in enumerate(layer_query_attns):
        ax = axes[layer_idx + 1]
        if len(query_attns) == 0:
            ax.axis('off')
            continue
        
        combined_attn = np.mean(query_attns, axis=0) if len(query_attns) > 1 else query_attns[0]
        
        # 使用改进的 overlay 函数 (sigma=5.0, alpha=0.7)
        overlaid_img = overlay_attention_on_image(img_np, combined_attn, alpha=0.7, sigma=5.0)
        
        ax.imshow(overlaid_img)
        ax.set_title(f'Layer {layer_idx}\n({len(query_attns)} queries avg)', fontsize=12)
        ax.axis('off')

    # 3. 最右侧：聚合热力图
    if aggregated_attn is not None:
        ax = axes[-1]
        # 聚合图通常更重要，alpha 设高一点，sigma 稍大一点使其平滑
        overlaid_agg = overlay_attention_on_image(img_np, aggregated_attn, alpha=0.75, sigma=5.0)
        ax.imshow(overlaid_agg)
        ax.set_title('Aggregated\n(All Layers)', fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def analyze_retrieval_errors(args, model, dataset, save_dir: str = "retrieval_errors"):
    """
    分析检索错误样本
    
    Args:
        args: 参数
        model: 模型
        dataset: 数据集
        save_dir: 保存目录
    """
    save_path = Path(args.save_dir) / save_dir
    save_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"开始分析检索错误样本，保存到 {save_path}")
    
    model.eval()
    
    # 提取数据库特征
    database_subset_ds = Subset(dataset, list(range(dataset.database_num)))
    database_dataloader = DataLoader(
        dataset=database_subset_ds, 
        num_workers=args.num_workers,
        batch_size=args.infer_batch_size, 
        pin_memory=(args.device=="cuda")
    )
    
    # 优化内存：分批处理，不保存所有图像
    database_features_list = []
    
    with torch.no_grad():
        for inputs, indices in tqdm(database_dataloader, desc="Extracting database features"):
            features = model(inputs.to(args.device))
            # 立即转换为numpy并释放GPU内存
            features_np = features.cpu().numpy()
            database_features_list.append(features_np)
            # 释放GPU内存
            del features, inputs
            torch.cuda.empty_cache() if args.device == "cuda" else None
    
    # 合并特征（分批合并以减少内存峰值）
    logging.info("合并数据库特征...")
    database_features = np.vstack(database_features_list)
    del database_features_list
    torch.cuda.empty_cache() if args.device == "cuda" else None
    
    # 确保features_dim已设置
    if not hasattr(args, 'features_dim'):
        args.features_dim = database_features.shape[1]
        logging.warning(f"自动设置 args.features_dim = {args.features_dim}")
    
    # 构建FAISS索引
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features.astype('float32'))
    
    # 分析每个查询文件夹
    error_samples = defaultdict(list)
    
    for folder_name, data in dataset.queries_data.items():
        logging.info(f"分析查询文件夹: {folder_name}")
        
        # 获取查询图像索引
        folder_paths = data['paths']
        start_idx = dataset.images_paths.index(folder_paths[0])
        end_idx = start_idx + len(folder_paths)
        
        queries_subset_ds = Subset(dataset, list(range(start_idx, end_idx)))
        queries_dataloader = DataLoader(
            dataset=queries_subset_ds, 
            num_workers=args.num_workers,
            batch_size=args.infer_batch_size, 
            pin_memory=(args.device=="cuda")
        )
        
        # 提取查询特征（优化内存：不保存所有图像，只保存错误样本的图像）
        query_features_list = []
        query_indices_list = []
        
        with torch.no_grad():
            for inputs, indices in tqdm(queries_dataloader, desc=f"Extracting query features for {folder_name}"):
                features = model(inputs.to(args.device))
                query_features_list.append(features.cpu().numpy())
                query_indices_list.extend((indices.numpy() - start_idx).tolist())
                # 释放GPU内存
                del features, inputs
                torch.cuda.empty_cache() if args.device == "cuda" else None
        
        query_features = np.vstack(query_features_list)
        del query_features_list
        torch.cuda.empty_cache() if args.device == "cuda" else None
        
        # 检索
        positives_per_query = data['soft_positives']
        k = max(args.recall_values)
        
        # 分批检索以减少内存峰值（对于大数据集）
        batch_size = 1000  # 每次处理1000个查询
        all_predictions = []
        all_distances = []
        
        for i in range(0, len(query_features), batch_size):
            end_idx = min(i + batch_size, len(query_features))
            batch_features = query_features[i:end_idx].astype('float32')
            distances_batch, predictions_batch = faiss_index.search(batch_features, k)
            all_predictions.append(predictions_batch)
            all_distances.append(distances_batch)
            del batch_features
            torch.cuda.empty_cache() if args.device == "cuda" else None
        
        predictions = np.vstack(all_predictions)
        distances = np.vstack(all_distances)
        del all_predictions, all_distances, query_features
        torch.cuda.empty_cache() if args.device == "cuda" else None
        
        # 找出错误样本（只保存必要信息，不保存图像）
        for query_idx, pred in enumerate(predictions):
            # 检查top-1是否正确
            if pred[0] not in positives_per_query[query_idx]:
                error_samples[folder_name].append({
                    'query_idx': query_idx,
                    'query_global_idx': start_idx + query_idx,
                    'predicted_idx': pred[0],
                    'top_k_predictions': pred[:k].tolist(),
                    'distances': distances[query_idx][:k].tolist(),
                    'positives': positives_per_query[query_idx],
                })
        
        del predictions, distances
        torch.cuda.empty_cache() if args.device == "cuda" else None
    
    # 保存错误样本信息
    error_info_file = save_path / "error_samples_info.txt"
    with open(error_info_file, 'w') as f:
        f.write(f"检索错误样本分析结果\n")
        f.write(f"数据集: {dataset.dataset_name}\n")
        f.write(f"=" * 80 + "\n\n")
        
        total_errors = 0
        for folder_name, errors in error_samples.items():
            f.write(f"查询文件夹: {folder_name}\n")
            f.write(f"错误样本数: {len(errors)}\n")
            total_errors += len(errors)
            
            for i, error in enumerate(errors[:10]):  # 只保存前10个
                f.write(f"\n错误样本 {i+1}:\n")
                f.write(f"  查询索引: {error['query_idx']}\n")
                f.write(f"  预测索引: {error['predicted_idx']}\n")
                f.write(f"  真实正样本: {error['positives'][:5]}\n")  # 只显示前5个
                f.write(f"  Top-5预测: {error['top_k_predictions'][:5]}\n")
                f.write(f"  Top-5距离: {[f'{d:.4f}' for d in error['distances'][:5]]}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
        
        f.write(f"总错误样本数: {total_errors}\n")
    
    logging.info(f"错误样本信息已保存到 {error_info_file}")
    logging.info(f"总错误样本数: {total_errors}")
    
    return error_samples


def visualize_error_samples(args, model, dataset, error_samples: Dict, 
                           save_dir: str = "error_visualization",
                           num_samples: int = 10):
    """
    可视化错误样本的attention热力图
    
    Args:
        args: 参数
        model: 模型
        dataset: 数据集
        error_samples: 错误样本字典
        save_dir: 保存目录
        num_samples: 每个文件夹要可视化的样本数量
    """
    save_path = Path(args.save_dir) / save_dir
    save_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"开始可视化错误样本，保存到 {save_path}")
    
    model.eval()
    model.module._return_stats = True
    
    for folder_name, errors in error_samples.items():
        if len(errors) == 0:
            continue
        
        folder_save_path = save_path / folder_name
        folder_save_path.mkdir(parents=True, exist_ok=True)
        
        # 只可视化前num_samples个错误样本
        for i, error in enumerate(errors[:num_samples]):
            query_idx = error['query_idx']
            query_global_idx = error['query_global_idx']
            
            # 获取查询图像
            try:
                dataset_item = dataset[query_global_idx]
                if isinstance(dataset_item, tuple):
                    query_image = dataset_item[0].unsqueeze(0)  # [1, C, H, W]
                else:
                    query_image = dataset_item.unsqueeze(0)
            except Exception as e:
                logging.warning(f"无法获取查询图像 {query_global_idx}: {e}")
                continue
            
            # 提取attention权重
            attention_weights_list, patch_grid_h, patch_grid_w = extract_attention_weights(
                model, query_image, args.device
            )
            
            if len(attention_weights_list) > 0:
                # 可视化叠加热力图
                save_file = folder_save_path / f"error_{i}_query_{query_idx}_stacked.png"
                visualize_stacked_heatmaps(
                    attention_weights_list,
                    query_image[0].cpu(),
                    str(save_file),
                    patch_grid_h=patch_grid_h,
                    patch_grid_w=patch_grid_w
                )
            
            if (i + 1) % 5 == 0:
                logging.info(f"已可视化 {i+1}/{min(num_samples, len(errors))} 个错误样本 ({folder_name})")


def compare_datasets_performance(args, model, dataset_names: List[str], 
                                 save_dir: str = "dataset_comparison"):
    """
    对比不同数据集的性能差异
    
    Args:
        args: 参数
        model: 模型
        dataset_names: 数据集名称列表
        save_dir: 保存目录
    """
    save_path = Path(args.save_dir) / save_dir
    save_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"开始对比数据集性能，保存到 {save_path}")
    
    results = {}
    
    for dataset_name in dataset_names:
        logging.info(f"测试数据集: {dataset_name}")
        
        # 加载数据集
        test_ds = datasets_ws.BaseDataset(
            args, args.eval_datasets_folder, dataset_name, "test"
        )
        
            # 运行测试
        try:
            all_recalls, combined_str = test.test(args, test_ds, model, args.test_method)
        except Exception as e:
            logging.error(f"测试数据集 {dataset_name} 时出错: {e}")
            import traceback
            logging.error(f"错误详情: {traceback.format_exc()}")
            # 即使测试失败，也保存错误信息
            results[dataset_name] = {
                'all_recalls': {},
                'combined_str': f"测试失败: {str(e)}",
                'error': str(e)
            }
            continue
        
        # 保存结果
        results[dataset_name] = {
            'all_recalls': all_recalls,
            'combined_str': combined_str
        }
        
        # 分析错误样本
        error_samples = analyze_retrieval_errors(
            args, model, test_ds, 
            save_dir=f"retrieval_errors_{dataset_name}"
        )
        
        # 可视化错误样本
        visualize_error_samples(
            args, model, test_ds, error_samples,
            save_dir=f"error_visualization_{dataset_name}",
            num_samples=10
        )
    
    # 保存对比结果
    comparison_file = save_path / "dataset_comparison.txt"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("数据集性能对比\n")
        f.write("=" * 80 + "\n\n")
        
        if len(results) == 0:
            f.write("警告: 没有成功测试的数据集\n")
            f.write("请检查错误日志以了解失败原因\n")
        else:
            for dataset_name, result in results.items():
                f.write(f"数据集: {dataset_name}\n")
                f.write(f"综合结果: {result['combined_str']}\n")
                
                if 'error' in result:
                    f.write(f"错误信息: {result['error']}\n")
                else:
                    if len(result['all_recalls']) > 0:
                        for folder_name, (_, recalls_str) in result['all_recalls'].items():
                            f.write(f"  {folder_name}: {recalls_str}\n")
                    else:
                        f.write("  无详细召回率信息\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
    
    logging.info(f"对比结果已保存到 {comparison_file}")
    
    return results


if __name__ == "__main__":
    from tqdm import tqdm
    
    # 解析参数
    args = parser.parse_arguments()
    
    # 设置日志
    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(join(args.save_dir, "error_analysis_log.txt"), mode='a')
        ]
    )
    
    commons.make_deterministic(args.seed)
    
    # 加载模型
    model = network.VPRNet(pretrained_foundation=True, foundation_model_path=args.foundation_model_path)
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)
    
    # 加载检查点
    if args.resume:
        checkpoint = torch.load(join(args.save_dir, "best_model.pth"), weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info("模型加载完成")
    
    # 设置必要的args属性（如果缺失）
    if not hasattr(args, 'features_dim'):
        args.features_dim = 4096
        logging.info(f"设置 args.features_dim = {args.features_dim}")
    
    if not hasattr(args, 'recall_values'):
        args.recall_values = [1, 5, 10, 20]
        logging.info(f"设置 args.recall_values = {args.recall_values}")
    
    if not hasattr(args, 'infer_batch_size'):
        args.infer_batch_size = 4
        logging.info(f"设置 args.infer_batch_size = {args.infer_batch_size}")
    
    if not hasattr(args, 'test_method'):
        args.test_method = "hard_resize"
        logging.info(f"设置 args.test_method = {args.test_method}")
    
    # 对比不同数据集
    dataset_names = ["amstertime/images", "Nordland/images_winter_as_quries"]  # 根据实际数据集名称调整
    
    # 检查数据集是否存在
    available_datasets = []
    for name in dataset_names:
        try:
            test_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, name, "test")
            available_datasets.append(name)
            logging.info(f"数据集 {name} 可用")
        except Exception as e:
            logging.warning(f"数据集 {name} 不可用: {e}")
    
    if len(available_datasets) > 0:
        results = compare_datasets_performance(args, model, available_datasets)
    else:
        logging.warning("没有可用的数据集，请检查数据集路径和名称")
