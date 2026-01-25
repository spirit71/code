"""
验证适配器(adapters)是否正常工作的辅助脚本
使用方法：在训练或推理代码中导入并调用验证函数
"""

import torch
from typing import List, Dict, Optional


def verify_adapters_in_forward(
    model,
    x_list: List[torch.Tensor],
    masks_list: List[torch.Tensor],
    verbose: bool = True
) -> Dict:
    """
    验证 forward_features_list 中适配器是否正常工作
    
    Args:
        model: Vision Transformer 模型实例
        x_list: 输入张量列表
        masks_list: 掩码列表
        verbose: 是否打印详细信息
    
    Returns:
        包含验证结果的字典
    """
    # 检查适配器是否存在
    has_adapters = hasattr(model, 'adapters') and model.adapters is not None
    adapter_count = len(model.adapters) if has_adapters else 0
    block_count = len(model.blocks) if hasattr(model, 'blocks') else 0
    
    results = {
        'has_adapters': has_adapters,
        'adapter_count': adapter_count,
        'block_count': block_count,
        'adapter_block_match': adapter_count == block_count,
        'warnings': []
    }
    
    if verbose:
        print("=" * 60)
        print("适配器验证报告")
        print("=" * 60)
        print(f"模型是否有adapters属性: {has_adapters}")
        print(f"适配器数量: {adapter_count}")
        print(f"Block数量: {block_count}")
        print(f"适配器与Block数量匹配: {results['adapter_block_match']}")
        
        if not has_adapters:
            print("\n⚠️  警告: 模型没有adapters属性或adapters为None")
            results['warnings'].append("模型没有adapters属性")
        elif not results['adapter_block_match']:
            print(f"\n⚠️  警告: 适配器数量({adapter_count})与Block数量({block_count})不匹配")
            results['warnings'].append(f"适配器数量与Block数量不匹配")
    
    # 运行前向传播（代码中已包含验证逻辑）
    try:
        with torch.no_grad():
            output = model.forward_features_list(x_list, masks_list)
        results['forward_success'] = True
        results['output_shape'] = {k: v.shape for k, v in output[0].items() if isinstance(v, torch.Tensor)}
    except Exception as e:
        results['forward_success'] = False
        results['error'] = str(e)
        if verbose:
            print(f"\n❌ 前向传播失败: {e}")
    
    return results


def compare_with_without_adapters(
    model,
    x_list: List[torch.Tensor],
    masks_list: List[torch.Tensor]
) -> Dict:
    """
    比较使用适配器和不使用适配器的输出差异
    
    Args:
        model: Vision Transformer 模型实例
        x_list: 输入张量列表
        masks_list: 掩码列表
    
    Returns:
        包含差异分析的字典
    """
    if not hasattr(model, 'adapters') or model.adapters is None:
        return {'error': '模型没有adapters属性'}
    
    # 保存原始适配器
    original_adapters = model.adapters
    
    # 运行带适配器的前向传播
    with torch.no_grad():
        output_with_adapters = model.forward_features_list(x_list, masks_list)
    
    # 临时禁用适配器（设置为None或恒等映射）
    model.adapters = None
    
    # 运行不带适配器的前向传播
    with torch.no_grad():
        output_without_adapters = model.forward_features_list(x_list, masks_list)
    
    # 恢复原始适配器
    model.adapters = original_adapters
    
    # 计算差异
    differences = {}
    for key in output_with_adapters[0].keys():
        if isinstance(output_with_adapters[0][key], torch.Tensor):
            diff = torch.norm(output_with_adapters[0][key] - output_without_adapters[0][key]).item()
            differences[key] = {
                'l2_norm_diff': diff,
                'max_diff': torch.max(torch.abs(output_with_adapters[0][key] - output_without_adapters[0][key])).item(),
                'mean_diff': torch.mean(torch.abs(output_with_adapters[0][key] - output_without_adapters[0][key])).item()
            }
    
    print("\n" + "=" * 60)
    print("适配器影响分析（有适配器 vs 无适配器）")
    print("=" * 60)
    for key, diff_info in differences.items():
        print(f"\n{key}:")
        print(f"  - L2范数差异: {diff_info['l2_norm_diff']:.6f}")
        print(f"  - 最大差异: {diff_info['max_diff']:.6f}")
        print(f"  - 平均差异: {diff_info['mean_diff']:.6f}")
    
    return {
        'differences': differences,
        'adapters_have_effect': any(d['l2_norm_diff'] > 1e-6 for d in differences.values())
    }


if __name__ == "__main__":
    print("这是一个验证辅助脚本，请在您的训练/测试代码中导入使用")
    print("\n使用示例:")
    print("""
    from verify_adapters import verify_adapters_in_forward, compare_with_without_adapters
    
    # 方法1: 基本验证（代码中已包含打印信息）
    results = verify_adapters_in_forward(model, x_list, masks_list)
    
    # 方法2: 比较有无适配器的差异
    diff_results = compare_with_without_adapters(model, x_list, masks_list)
    """)





