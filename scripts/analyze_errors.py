"""
错误样本分析和热力图导出工具
用于分析模型预测错误的样本，并导出attention热力图
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

from torch.utils.data import DataLoader
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
    提取模型的attention权重
    
    Args:
        model: 模型
        images: 输入图像 [B, C, H, W]
        device: 设备
        patch_grid_h: patch网格高度
        patch_grid_w: patch网格宽度
    
    Returns:
        attention_weights: List of attention weights for each layer
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
            return attention_weights
    return []


def visualize_attention_heatmap(attention_weights: torch.Tensor, 
                                image: torch.Tensor,
                                save_path: str,
                                patch_grid_h: int = 16,
                                patch_grid_w: int = 16,
                                query_idx: int = 0):
    """
    可视化attention热力图
    
    Args:
        attention_weights: [num_queries, num_keys] attention权重
        image: 原始图像 [C, H, W]
        save_path: 保存路径
        patch_grid_h: patch网格高度
        patch_grid_w: patch网格宽度
        query_idx: 要可视化的query索引
    """
    # 获取指定query的attention权重
    if attention_weights.dim() == 2:
        attn = attention_weights[query_idx].numpy()  # [num_keys]
    else:
        attn = attention_weights.numpy()
    
    # 如果包含cls token，只取patch tokens部分
    if len(attn) > patch_grid_h * patch_grid_w:
        attn = attn[-patch_grid_h * patch_grid_w:]  # 只取patch tokens
    
    # 重塑为网格形状
    attn_grid = attn.reshape(patch_grid_h, patch_grid_w)
    
    # 创建图像
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # 原始图像
    img_np = image.permute(1, 2, 0).cpu().numpy()
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention热力图
    im = axes[1].imshow(attn_grid, cmap='hot', interpolation='nearest')
    axes[1].set_title(f'Attention Heatmap (Query {query_idx})')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_error_samples(args, model, dataset, num_samples: int = 10, 
                          save_dir: str = "error_analysis"):
    """
    分析错误样本
    
    Args:
        args: 参数
        model: 模型
        dataset: 数据集
        num_samples: 要分析的样本数量
        save_dir: 保存目录
    """
    save_path = Path(args.save_dir) / save_dir
    save_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"开始分析错误样本，保存到 {save_path}")
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # 获取所有查询和数据库特征
    model.eval()
    all_queries = []
    all_db = []
    all_query_labels = []
    all_db_labels = []
    all_images = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.view(-1, *images.shape[2:]).to(args.device)
            descriptors = model(images)
            all_images.append(images.cpu())
            
            # 分离查询和数据库
            # 这里需要根据实际数据集结构调整
            # 假设前N个是查询，后面是数据库
            # 实际使用时需要根据数据集的具体结构调整
    
    # 计算相似度并找出错误样本
    # 这里简化处理，实际需要根据测试逻辑实现
    logging.info("错误样本分析完成")


def export_attention_heatmaps(args, model, dataset, num_samples: int = 20,
                              save_dir: str = "attention_heatmaps"):
    """
    导出attention热力图
    
    Args:
        args: 参数
        model: 模型
        dataset: 数据集
        num_samples: 要导出的样本数量
        save_dir: 保存目录
    """
    save_path = Path(args.save_dir) / save_dir
    save_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"开始导出attention热力图，保存到 {save_path}")
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    model.eval()
    count = 0
    
    for images, labels in dataloader:
        if count >= num_samples:
            break
        
        # === 修改开始 ===
        # 直接传送到设备，不要用 .view() 改变形状
        # DataLoader 出来的默认就是 [B, C, H, W]，这正是模型需要的
        images = images.to(args.device)
        # === 修改结束 ===
        # 提取attention权重
        attention_weights_list = extract_attention_weights(
            model, images, args.device, patch_grid_h=16, patch_grid_w=16
        )
        
        if len(attention_weights_list) > 0:
            # 为每个layer和每个query保存热力图
            for layer_idx, attn_weights in enumerate(attention_weights_list):
                if attn_weights is not None:
                    num_queries = attn_weights.shape[0] if attn_weights.dim() == 2 else 1
                    
                    # 只保存前几个query的热力图
                    for query_idx in range(min(4, num_queries)):
                        save_file = save_path / f"sample_{count}_layer_{layer_idx}_query_{query_idx}.png"
                        visualize_attention_heatmap(
                            attn_weights,
                            images[0].cpu(),
                            str(save_file),
                            patch_grid_h=16,
                            patch_grid_w=16,
                            query_idx=query_idx
                        )
        
        count += 1
        if count % 5 == 0:
            logging.info(f"已处理 {count}/{num_samples} 个样本")
    
    logging.info(f"Attention热力图导出完成，共 {count} 个样本")


if __name__ == "__main__":
    # 解析参数
    args = parser.parse_arguments()
    
    # 设置日志
    # commons.setup_logging(args.save_dir)
    # === 新增代码开始 ===
    import os
    # 如果文件夹不存在则创建，存在则不做任何事
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 手动配置日志，允许在现有文件夹中写入
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),  # 输出到控制台
            logging.FileHandler(join(args.save_dir, "analysis_log.txt"), mode='a') # 输出到文件
        ]
    )
# === 新增代码结束 ===
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
    
    # 加载数据集
    val_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "val")
    
    # 导出attention热力图
    export_attention_heatmaps(args, model, val_ds, num_samples=20)
    
    # 分析错误样本（需要根据实际测试逻辑实现）
    # analyze_error_samples(args, model, val_ds, num_samples=10)
