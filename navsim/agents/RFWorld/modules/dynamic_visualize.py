import os
import torch
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm


def visualize_lidar_feature(lidar_feature, save_path: str):
    """
    可视化 [256, 256] 的雷达 BEV 特征（2D柱状密度图）

    :param lidar_feature: 形状可以是 [256, 256], [1, 256, 256] 或 [2, 256, 256] 的 Tensor/Array
    :param save_path: 保存图像的绝对或相对路径 (如 './docs/debug_lidar_0.png')
    """
    # 1. 数据类型转换：如果是 PyTorch Tensor，转换为 numpy 数组
    if isinstance(lidar_feature, torch.Tensor):
        lidar_feature = lidar_feature.detach().cpu().numpy()

    # 2. 维度处理
    if lidar_feature.ndim == 3:
        if lidar_feature.shape[0] == 1:
            # 单通道（仅包含地面以上的点云）
            feature_2d = lidar_feature[0]
        elif lidar_feature.shape[0] == 2:
            # 双通道（地面、地面以上），取最大值或求和融合显示
            feature_2d = np.max(lidar_feature, axis=0)
        else:
            # 兼容其他通道数
            feature_2d = lidar_feature[0]
    elif lidar_feature.ndim == 2:
        feature_2d = lidar_feature
    else:
        raise ValueError(f"不支持的 lidar_feature 维度: {lidar_feature.shape}，期望二维或三维。")

    # 3. 创建画布
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')

    # 4. 绘制热力图
    # magma colormap：0 对应黑色，数值越高越亮(红/黄/白)，非常适合呈现稀疏点云密度
    im = ax.imshow(feature_2d, cmap='magma', origin='upper', interpolation='none')

    # 添加颜色条 (Colorbar)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Point Density', rotation=270, labelpad=15)

    # 5. 绘制自车 (Ego Vehicle) 标志
    # 假设尺寸为 256x256，自车默认位于网格正中心位置
    # 注：如果你配置的 LiDAR 范围(lidar_min_x, max_x)不是前后对称的，请适当调整 128 这个坐标
    ego_x, ego_y = 128, 128
    ax.plot(ego_x, ego_y, 'r^', markersize=12, label='Ego Vehicle')

    # 6. 设置图表细节
    ax.set_title(f'LiDAR BEV Feature Map ({feature_2d.shape[0]}x{feature_2d.shape[1]})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Y axis (pixels)', fontsize=12)
    ax.set_ylabel('X axis (pixels)', fontsize=12)
    ax.grid(False)  # 关闭网格以突出点云
    ax.legend(loc='upper right')

    # 7. 保存图像
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[DEBUG] LiDAR feature visualization saved to: {save_path}")


def visualize_bev_semantic_map(semantic_map: Union[torch.Tensor, np.ndarray], save_path: str):
    """
    可视化 BEV 语义分割图

    :param semantic_map: 形状可以是 [H, W] (整数标签) 或 [C, H, W] (模型输出概率) 的 Tensor/Array
    :param save_path: 保存图像的路径
    """
    if isinstance(semantic_map, torch.Tensor):
        semantic_map = semantic_map.detach().cpu().numpy()

    # 处理维度：如果是 [C, H, W]，通常是模型输出的 One-hot 或 Logits，需要 argmax 取类别
    if semantic_map.ndim == 3:
        if semantic_map.shape[0] > 1:  # [Classes, H, W]
            semantic_map = np.argmax(semantic_map, axis=0)
        else:  # [1, H, W]
            semantic_map = semantic_map[0]

    # 确保是整数类型
    semantic_map = semantic_map.astype(np.int32)

    # --- 类别与颜色定义 (参考 default.py 的 bev_semantic_classes) ---
    # 0: 背景, 1: 道路, 2: 人行道, 3: 车道中心线, 4: 静态物体, 5: 车辆, 6: 行人, 7: 自车
    CLASS_LABELS = {
        0: "Background",
        1: "Road (Drivable)",
        2: "Walkways",
        3: "Centerline",
        4: "Static Objects",
        5: "Vehicles",
        6: "Pedestrians",
        7: "Ego Vehicle"
    }

    # 为 8 个类别定义美观且对比鲜明的 RGB 颜色 (0.0 ~ 1.0)
    COLORS = [
        (0.00, 0.00, 0.00),  # 0: 纯黑 (Background)
        (0.30, 0.30, 0.35),  # 1: 深灰蓝 (Road)
        (0.45, 0.45, 0.40),  # 2: 卡其灰 (Walkways)
        (0.90, 0.80, 0.10),  # 3: 亮黄 (Centerline)
        (0.80, 0.40, 0.10),  # 4: 橙棕色 (Static Objects - barrier, cone etc.)
        (0.20, 0.50, 0.90),  # 5: 亮蓝色 (Vehicles)
        (0.90, 0.20, 0.20),  # 6: 红色 (Pedestrians)
        (0.10, 0.85, 0.30),  # 7: 亮绿色 (Ego Vehicle)
    ]

    cmap = ListedColormap(COLORS)
    # BoundaryNorm 确保 0->颜色0，1->颜色1，离散映射
    bounds = np.arange(-0.5, 8.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    im = ax.imshow(semantic_map, cmap=cmap, norm=norm, origin='upper', interpolation='nearest')

    # --- 自定义图例 (Legend) ---
    # 生成图例色块
    patches = [mpatches.Patch(color=COLORS[i], label=f"{i}: {CLASS_LABELS[i]}") for i in range(len(COLORS))]
    # 放置图例在图像外部右侧
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)

    # --- 坐标轴设置 ---
    ax.set_title(f'BEV Semantic Map ({semantic_map.shape[0]}x{semantic_map.shape[1]})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Y axis (pixels)', fontsize=12)
    ax.set_ylabel('X axis (pixels)', fontsize=12)
    ax.grid(False)

    # --- 保存 ---
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[DEBUG] BEV Semantic Map saved to: {save_path}")


def visualize_fov_mask(mask: Union[torch.Tensor, np.ndarray], valid_ratio: float, save_path: str):
    """
    可视化未来视野重叠掩码 (FOV Mask)，并在图像上标注有效比例。

    :param mask: 形状 [H, W] 的布尔型或整型 Tensor/Array (1表示有效, 0表示无效)
    :param valid_ratio: 掩码的有效比例 (float)
    :param save_path: 保存图像的路径
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # 处理维度
    if mask.ndim == 3:
        mask = mask[0]

    mask = mask.astype(np.float32)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')

    # 绘制二值掩码，0 为深色(无效区)，1 为青色(有效区)
    im = ax.imshow(mask, cmap='bone', origin='upper', interpolation='none', vmin=0, vmax=1)

    # 在左上角添加 Valid Ratio 文本标注
    ax.text(0.03, 0.97, f'Valid Ratio: {valid_ratio:.2%}',
            transform=ax.transAxes, fontsize=14, color='white',
            verticalalignment='top', fontweight='bold',
            bbox=dict(facecolor='red', alpha=0.7, edgecolor='none', pad=5))

    # 绘制自车中心标识
    ego_x, ego_y = 128, 128
    ax.plot(ego_x, ego_y, 'r^', markersize=12, label='Ego (Current Frame)')

    ax.set_title(f'Future FOV Mask ({mask.shape[0]}x{mask.shape[1]})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Y axis (pixels)', fontsize=12)
    ax.set_ylabel('X axis (pixels)', fontsize=12)
    ax.grid(False)
    ax.legend(loc='upper right')

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[DEBUG] FOV Mask saved to: {save_path}")


# ==========================================
# 测试代码 (如果在终端直接运行该文件)
# ==========================================
if __name__ == "__main__":
    # 模拟一个 256x256 的雷达特征张量，中间有一条“道路”高密度区域
    dummy_feature = np.zeros((256, 256))
    dummy_feature[50:200, 120:136] = 0.8  # 模拟正前方的车道
    dummy_feature[100, 128] = 1.0  # 模拟前方的一个障碍物

    save_dir = "./docs/"
    visualize_lidar_feature(dummy_feature, os.path.join(save_dir, "sample_lidar_feature.png"))