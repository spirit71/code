import torch
import torch.nn as nn
import numpy as np


class GeoVisualGraphSampler:
    """
    Online Geo-Visual Graph Construction and Greedy Weighted Sampling
    From SAGE paper
    
    Paper: SAGE - Spatial-visual Adaptive Graph Exploration for Efficient VPR
    Section 3.3 & 3.4: Online Graph Creation & Greedy Weighted Sampling
    
    原理：
    1. 每个epoch重建地理-视觉亲和图
    2. 亲和度: Wij = -(d_geo(i,j) · d_vis(i,j))
    3. 贪心扩展: 从高亲和度seed迭代添加节点
    
    作用：聚焦训练在最具信息量的困难样本上
    """
    def __init__(self, geo_threshold=25.0, affinity_threshold=-2.88e3, 
                 num_places=15, clique_size=4):
        self.geo_threshold = geo_threshold
        self.affinity_threshold = affinity_threshold
        self.num_places = num_places
        self.clique_size = clique_size
    
    def compute_affinity_matrix(self, descriptors, geo_coords=None):
        """
        计算亲和度矩阵
        
        Args:
            descriptors: [N, D] - image descriptors
            geo_coords: [N, 2] - (lat, lon) coordinates, optional
        Returns:
            W: [N, N] - affinity matrix
        """
        N = descriptors.shape[0]
        
        vis_dist = torch.cdist(descriptors, descriptors, p=2)
        
        if geo_coords is not None:
            geo_dist = torch.cdist(geo_coords, geo_coords, p=2)
            W = -geo_dist * vis_dist
        else:
            W = -vis_dist
        
        W.fill_diagonal_(0)
        
        return W
    
    def compute_seed_scores(self, W):
        """
        计算每个节点的seed score
        S(i) = (1/(N-1)) * Σ Wij
        
        Args:
            W: [N, N] - affinity matrix
        Returns:
            seed_scores: [N] - seed scores for each node
        """
        N = W.shape[0]
        seed_scores = W.sum(dim=1) / (N - 1)
        return seed_scores
    
    def greedy_clique_expansion(self, W, min_size=4, max_size=10):
        """
        贪心加权Clique扩展采样
        
        1. 选择seed score最高的节点作为初始anchor
        2. 迭代添加与当前clique平均亲和度最高的节点
        3. 直到达到目标大小
        
        Args:
            W: [N, N] - affinity matrix
            min_size: 最小clique大小
            max_size: 最大clique大小
        Returns:
            clique: [k] - 采样的节点索引列表
        """
        N = W.shape[0]
        
        seed_scores = self.compute_seed_scores(W)
        seed_idx = torch.argmax(seed_scores)
        clique = [seed_idx.item()]
        
        for _ in range(max_size - 1):
            candidates = [i for i in range(N) if i not in clique]
            if not candidates:
                break
            
            best_candidate = None
            best_avg_affinity = -float('inf')
            
            for c in candidates:
                avg_affinity = W[clique, c].mean().item()
                if avg_affinity > best_avg_affinity:
                    best_avg_affinity = avg_affinity
                    best_candidate = c
            
            if best_candidate is not None and len(clique) < self.clique_size:
                clique.append(best_candidate)
            else:
                break
        
        if len(clique) < min_size:
            return None
        
        return clique
    
    def sample_batch(self, descriptors, geo_coords=None, labels=None):
        """
        批量采样接口
        
        Args:
            descriptors: [N, D]
            geo_coords: [N, 2], optional
            labels: [N], optional for label filtering
        Returns:
            sampled_indices: [k] - 采样的索引
        """
        W = self.compute_affinity_matrix(descriptors, geo_coords)
        clique = self.greedy_clique_expansion(W)
        
        if clique is None:
            indices = torch.randperm(len(descriptors))[:self.clique_size]
            return indices.tolist()
        
        return clique
    
    def build_sparse_graph(self, W, threshold=None):
        """
        构建稀疏亲和图
        
        Args:
            W: [N, N] - affinity matrix
            threshold: 边阈值
        Returns:
            edges: [(i, j, w), ...] - 边列表
        """
        if threshold is None:
            threshold = self.affinity_threshold
        
        edges = []
        N = W.shape[0]
        for i in range(N):
            for j in range(i + 1, N):
                if W[i, j] > threshold:
                    edges.append((i, j, W[i, j].item()))
        
        return edges


class GeoVisualGraphSamplerV2(GeoVisualGraphSampler):
    """
    改进版GeoVisualGraphSampler，支持更灵活的采样策略
    """
    def __init__(self, geo_threshold=25.0, affinity_threshold=-2.88e3, 
                 num_places=15, clique_size=4, temperature=1.0):
        super().__init__(geo_threshold, affinity_threshold, num_places, clique_size)
        self.temperature = temperature
    
    def probabilistic_sampling(self, W, num_samples=4):
        """
        概率采样 - 基于亲和度分布采样
        
        Args:
            W: [N, N] - affinity matrix
            num_samples: 采样数量
        Returns:
            sampled: [num_samples] - 采样索引
        """
        N = W.shape[0]
        
        seed_scores = self.compute_seed_scores(W)
        probs = torch.softmax(seed_scores / self.temperature, dim=0)
        
        sampled_indices = torch.multinomial(probs, num_samples, replacement=False)
        
        return sampled_indices.tolist()
    
    def balanced_sampling(self, W, labels, num_per_class=2):
        """
        平衡采样 - 确保每个类别都有样本
        
        Args:
            W: [N, N]
            labels: [N]
            num_per_class: 每个类别采样的数量
        Returns:
            sampled: 采样索引
        """
        unique_labels = torch.unique(labels)
        sampled = []
        
        for label in unique_labels:
            mask = labels == label
            indices = torch.where(mask)[0]
            
            if len(indices) <= num_per_class:
                sampled.extend(indices.tolist())
            else:
                clique = self.greedy_clique_expansion(W[indices][:, indices])
                if clique:
                    sampled.extend([indices[i] for i in clique[:num_per_class]])
                else:
                    sampled.extend(indices[:num_per_class].tolist())
        
        return sampled
