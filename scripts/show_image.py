# -*- coding: utf-8 -*-
"""
图片选择与显示脚本
功能：从指定文件夹中按排序方式选出第 N 张图片并显示
作者：辅助小助手
"""

import os
from pathlib import Path
from PIL import Image  # 需安装：pip install Pillow

def get_image_files(folder_path):
    """获取文件夹中所有常见格式的图片文件"""
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")
    
    # 支持的图片格式（不区分大小写）
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    image_files = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        raise ValueError(f"文件夹中未找到图片文件: {folder_path}")
    
    return image_files

def sort_images(image_files, sort_by='name'):
    """
    对图片列表排序
    sort_by 可选值:
        'name'   - 按文件名升序（默认）
        'time'   - 按修改时间升序（最早→最新）
        'rtime'  - 按修改时间降序（最新→最早）
    """
    if sort_by == 'name':
        return sorted(image_files, key=lambda x: x.name.lower())
    elif sort_by == 'time':
        return sorted(image_files, key=lambda x: x.stat().st_mtime)
    elif sort_by == 'rtime':
        return sorted(image_files, key=lambda x: x.stat().st_mtime, reverse=True)
    else:
        raise ValueError("sort_by 参数错误，应为 'name'、'time' 或 'rtime'")

def show_image_at_index(folder_path, index, sort_by='name'):
    """
    显示指定序号的图片（序号从1开始）
    参数:
        folder_path: 图片文件夹路径，例如 r"D:\my_images"
        index: 要显示的图片序号（从1开始），例如 143
        sort_by: 排序方式，'name'（默认）、'time'、'rtime'
    """
    try:
        # 获取并排序图片
        images = get_image_files(folder_path)
        sorted_images = sort_images(images, sort_by)
        
        total = len(sorted_images)
        print(f"✓ 共找到 {total} 张图片")
        
        # 检查序号是否合法
        if index < 1 or index > total:
            print(f"✗ 错误：序号 {index} 超出范围（有效范围: 1 ~ {total}）")
            return
        
        # 获取目标图片
        target = sorted_images[index - 1]  # Python索引从0开始
        print(f"✓ 正在显示第 {index} 张图片: {target.name}")
        print(f"  完整路径: {target}")
        
        # 显示图片
        img = Image.open(target)
        img.show()  # 调用系统默认图片查看器打开
        
        # 打印图片基本信息
        print(f"  尺寸: {img.width} × {img.height}")
        print(f"  格式: {img.format}, 模式: {img.mode}")
        
    except Exception as e:
        print(f"✗ 出错: {e}")

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 【请修改以下两行参数】
    FOLDER = r"/home/code_qy_7_28/VPR-datasets-downloader/datasets/Nordland/images_winter_as_quries/test/queries"  # ← 替换为你的图片文件夹路径（注意用 r 前缀或双反斜杠）
    TARGET_INDEX = 604                # ← 想查看第几张？（从1开始）
    
    # 排序方式（三选一）：
    # 'name'  - 按文件名排序（推荐，稳定）
    # 'time'  - 按修改时间升序（最早→最新）
    # 'rtime' - 按修改时间降序（最新→最早）
    SORT_METHOD = 'name'
    
    # 执行显示
    show_image_at_index(FOLDER, TARGET_INDEX, SORT_METHOD)