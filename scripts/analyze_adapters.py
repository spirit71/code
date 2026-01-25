import matplotlib.pyplot as plt
import numpy as np
import logging # 引入日志模块

def analyze_adapters(model, epoch, writer=None):
    # 1. 获取日志记录器（它会自动继承主程序中的配置）
    logger = logging.getLogger(__name__)

    # 2. 处理 DataParallel / 结构路径提取
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model

    # 路径匹配：VPRNet -> backbone -> adapters
    if hasattr(actual_model, 'backbone') and hasattr(actual_model.backbone, 'adapters'):
        target_adapters = actual_model.backbone.adapters
    elif hasattr(actual_model, 'adapters'):
        target_adapters = actual_model.adapters
    else:
        logger.error(f"无法找到 adapters。当前模块包含: {list(actual_model._modules.keys())}")
        return

    # 3. 提取数据
    alphas = [adapter.alpha.item() for adapter in target_adapters]
    layers = list(range(len(alphas)))
    max_idx = np.argmax(np.abs(alphas))
    alpha_mean = sum(alphas) / len(alphas)

    # --- 4. 导出到日志文件 ---
    logger.info("-" * 30)
    logger.info(f"Epoch {epoch:03d} Adapter Analysis:")
    logger.info(f"  > Alpha Mean (整体微调强度): {alpha_mean:.6f}")
    logger.info(f"  > 最活跃层: Index {max_idx} (强度: {alphas[max_idx]:.4f})")

    # 启发性逻辑导出
    if max_idx > len(alphas) * 0.7:
        insight = "结论：模型侧重于『语义适配』，DINOv2 底层特征非常稳健。"
    elif max_idx < len(alphas) * 0.3:
        insight = "结论：模型侧重于『基础特征重构』，数据域差异较大。"
    else:
        insight = "结论：微调强度分布在中间层，平衡了基础特征与语义逻辑。"
    
    logger.info(f"  > 启发性发现: {insight}")
    logger.info("-" * 30)

    # --- 5. TensorBoard 与 绘图逻辑 ---
    plt.figure(figsize=(10, 4))
    plt.bar(layers, alphas, color='skyblue')
    plt.xlabel('Transformer Block Index')
    plt.ylabel('Alpha Value')
    plt.title(f'Epoch {epoch}: Adapter Strength')
    plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5)

    if writer:
        writer.add_figure('Adapter/Alpha_Distribution', plt.gcf(), epoch)
        for i, val in enumerate(alphas):
            writer.add_scalar(f'Adapter/Alpha_Layer_{i}', val, epoch)
    
    plt.close()