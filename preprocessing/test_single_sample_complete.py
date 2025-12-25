"""
Single Sample Complete Test: YOLO Mask + PLY Point Cloud + Depth Map Visualization
"""
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
import json
import open3d as o3d

# 加载配置
CONFIG_FILE = Path("config.json")
with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    config = json.load(f)

ORGANIZED_DIR = Path(config['paths']['organized_dir'])
OUTPUT_SIZE = tuple(config['processing']['output_size'])
YOLO_MODEL_PATH = config.get('yolo', {}).get('model_path', r"E:\ECSF\dahuangyu\code\runs\segment\train4\weights\best.pt")
YOLO_CONF_THRES = config.get('yolo', {}).get('conf_thres', 0.5)

# ========== 选择测试样本 ==========
sample_dir = ORGANIZED_DIR / "wild" / "sample_0001"
# sample_dir = ORGANIZED_DIR / "farmed" / "sample_0001"

print("=" * 60)
print("Single Sample Complete Test")
print("=" * 60)
print(f"Test Sample: {sample_dir}")

# ========== 1. 加载RGB图像 ==========
rgb_file = sample_dir / "rgb.png"
if not rgb_file.exists():
    print(f"错误：未找到RGB文件 {rgb_file}")
    exit(1)

rgb_img = cv2.imread(str(rgb_file))
rgb_img_rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
original_size = (rgb_img.shape[1], rgb_img.shape[0])  # (W, H)

print(f"✓ RGB image loaded: {rgb_img.shape}")
print(f"  Image size: {rgb_img.shape[1]} x {rgb_img.shape[0]} (W x H)")

# ========== 2. 使用YOLO生成Mask ==========
print(f"\n加载YOLO模型: {YOLO_MODEL_PATH}")
model = YOLO(YOLO_MODEL_PATH)

print(f"使用YOLO生成mask（置信度阈值: {YOLO_CONF_THRES}）...")
results = model.predict(rgb_img, conf=YOLO_CONF_THRES, verbose=False)
result = results[0]

yolo_mask = None
if result.masks is not None and len(result.masks) > 0:
    boxes = result.boxes
    if len(boxes) > 0:
        confs = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))
        cls_id = int(boxes.cls[best_idx].cpu().numpy())
        best_conf = float(confs[best_idx])
        cls_name = result.names[cls_id]
        
        print(f"✓ Detected target: {cls_name}, confidence: {best_conf:.3f}")
        
        # 获取mask
        mask = result.masks[best_idx].data.cpu().numpy()
        if mask.ndim == 3:
            mask = mask[0]
        yolo_mask = (mask * 255).astype(np.uint8)
        
        # 确保尺寸匹配
        if yolo_mask.shape != rgb_img.shape[:2]:
            yolo_mask = cv2.resize(yolo_mask, (rgb_img.shape[1], rgb_img.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
        
        print(f"✓ YOLO Mask generated: {yolo_mask.shape}, foreground: {(yolo_mask > 0).sum() / yolo_mask.size * 100:.1f}%")
    else:
        print("警告：检测到mask但没有box信息")
        yolo_mask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
else:
    print("警告：YOLO未检测到目标，使用全0mask")
    yolo_mask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)

# ========== 3. 加载PLY点云 ==========
ply_file = sample_dir / "pointcloud.ply"
if not ply_file.exists():
    print(f"错误：未找到PLY文件 {ply_file}")
    exit(1)

print(f"\nLoading PLY point cloud: {ply_file}")
pcd = o3d.io.read_point_cloud(str(ply_file))
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors) if pcd.has_colors() else None

print(f"✓ Point cloud loaded: {len(points)} points")
print(f"  Point cloud range:")
print(f"    X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
print(f"    Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
print(f"    Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

# ========== 4. 从有序点云直接提取深度图 ==========
def extract_depth_from_organized_pointcloud(points, image_size):
    """
    Extract depth map from organized point cloud (Zivid相机输出)
    
    Zivid相机输出的PLY是有序点云，第N个点严格对应图像第N个像素
    直接reshape即可完美对齐，0偏差！
    
    Args:
        points: Point cloud (N, 3) - 有序点云
        image_size: (W, H) target image size
    
    Returns:
        depth_map: Depth map (H, W)
        is_organized: Whether point cloud is organized
    """
    W, H = image_size
    expected_points = H * W
    actual_points = len(points)
    
    print(f"  Image resolution: {W} x {H}")
    print(f"  Expected points: {expected_points}")
    print(f"  Actual points: {actual_points}")
    
    if actual_points == expected_points:
        print("  ✓ Point count matches perfectly! Using organized point cloud method.")
        # 直接reshape成(H, W, 3)
        grid_points = points.reshape(H, W, 3)
        # 提取深度(Z通道)
        depth_map = grid_points[:, :, 2].copy()
        # 处理无效值
        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
        return depth_map, True
    else:
        print(f"  ⚠️ Point count mismatch! Using projection method (may have alignment errors).")
        # 回退到投影方法（但会有偏差）
        return None, False

print("\nExtracting depth map from organized point cloud...")

# 尝试从有序点云直接提取深度图
depth_map, is_organized = extract_depth_from_organized_pointcloud(points, original_size)

if not is_organized:
    # 如果不是有序点云，回退到投影方法（会有偏差）
    print("  ⚠️ Falling back to projection method (may have alignment errors)")
    # 简单的投影方法（作为备选）
    valid_mask = np.isfinite(points).all(axis=1)
    points_valid = points[valid_mask]
    
    if len(points_valid) > 0:
        min_xy = points_valid[:, :2].min(axis=0)
        max_xy = points_valid[:, :2].max(axis=0)
        scale = np.min(original_size / (max_xy - min_xy + 1e-6))
        offset = (original_size - (max_xy - min_xy) * scale) / 2
        
        xy = (points_valid[:, :2] - min_xy) * scale + offset
        u = np.round(xy[:, 0]).astype(int)
        v = np.round(xy[:, 1]).astype(int)
        z = points_valid[:, 2]
        
        depth_map = np.zeros(original_size[::-1], dtype=np.float32)
        valid_pixels = (u >= 0) & (u < original_size[0]) & (v >= 0) & (v < original_size[1])
        u = u[valid_pixels]
        v = v[valid_pixels]
        z = z[valid_pixels]
        
        for i in range(len(u)):
            if depth_map[v[i], u[i]] == 0 or z[i] < depth_map[v[i], u[i]]:
                depth_map[v[i], u[i]] = z[i]
    else:
        depth_map = None

if depth_map is not None:
    valid_mask = np.isfinite(depth_map) & (depth_map > 0)
    depth_min = depth_map[valid_mask].min() if valid_mask.sum() > 0 else 0
    depth_max = depth_map[valid_mask].max() if valid_mask.sum() > 0 else 0
    print(f"✓ 深度图生成成功: {depth_map.shape}")
    print(f"  有效像素: {valid_mask.sum()}")
    print(f"  深度范围: [{depth_min:.3f}, {depth_max:.3f}]")
else:
    print("错误：无法生成深度图")
    exit(1)

# ========== 5. 根据Mask裁剪点云 ==========
print("\nFiltering point cloud using YOLO Mask...")

if is_organized:
    # 有序点云：直接使用mask索引
    print("  Using organized point cloud method (perfect alignment)")
    # Reshape点云到图像网格
    W, H = original_size
    grid_points = points.reshape(H, W, 3)
    
    # 使用mask直接索引
    mask_binary = yolo_mask > 0
    # 确保mask尺寸匹配
    if mask_binary.shape != (H, W):
        mask_binary = cv2.resize(mask_binary.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    
    # 提取mask内的点
    points_masked = grid_points[mask_binary]
    points_masked = points_masked.reshape(-1, 3)
    
    print(f"  ✓ Organized point cloud: {len(points)} total points")
    print(f"  ✓ Masked points: {len(points_masked)} ({len(points_masked)/len(points)*100:.1f}%)")
else:
    # 非有序点云：使用投影方法（会有偏差）
    print("  Using projection method (may have alignment errors)")
    valid_mask = np.isfinite(points).all(axis=1)
    points_valid = points[valid_mask]
    
    if len(points_valid) > 0:
        min_xy = points_valid[:, :2].min(axis=0)
        max_xy = points_valid[:, :2].max(axis=0)
        scale = np.min(original_size / (max_xy - min_xy + 1e-6))
        offset = (original_size - (max_xy - min_xy) * scale) / 2
        
        xy = (points_valid[:, :2] - min_xy) * scale + offset
        u = np.round(xy[:, 0]).astype(int)
        v = np.round(xy[:, 1]).astype(int)
        
        valid_pixels = (u >= 0) & (u < original_size[0]) & (v >= 0) & (v < original_size[1])
        u = u[valid_pixels]
        v = v[valid_pixels]
        points_valid = points_valid[valid_pixels]
        
        mask_values = yolo_mask[v, u] > 0
        points_masked = points_valid[mask_values]
    else:
        points_masked = points

print(f"✓ Original point cloud: {len(points)} points")
if is_organized:
    print(f"✓ Masked points: {len(points_masked)} ({len(points_masked)/len(points)*100:.1f}%)")
else:
    print(f"✓ Valid projected points: {len(points_valid)} points")
    print(f"✓ Masked points: {len(points_masked)} ({len(points_masked)/len(points)*100:.1f}%)")

# 创建裁剪后的点云对象
pcd_masked = o3d.geometry.PointCloud()
pcd_masked.points = o3d.utility.Vector3dVector(points_masked)
if colors is not None:
    if is_organized:
        # 有序点云：直接使用mask索引颜色
        grid_colors = colors.reshape(H, W, 3)
        colors_masked = grid_colors[mask_binary]
        colors_masked = colors_masked.reshape(-1, 3)
    else:
        # 非有序点云：使用投影方法
        colors_valid = colors[valid_mask]
        colors_valid = colors_valid[valid_pixels]
        colors_masked = colors_valid[mask_values]
    pcd_masked.colors = o3d.utility.Vector3dVector(colors_masked)

# ========== 6. 裁剪数据（根据Mask，扩大边界框） ==========
print("\nCropping data based on mask (with expansion)...")

# 计算mask的边界框
mask_binary = yolo_mask > 0
if mask_binary.sum() == 0:
    print("  ⚠️ Warning: Mask is empty, cannot crop")
    bbox = None
else:
    # 找到mask的边界
    coords = np.column_stack(np.where(mask_binary))
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # 扩大边界框（扩大30%，确保不丢失数据）
    H_img, W_img = rgb_img.shape[:2]
    expand_ratio = 0.1  # 30%外扩，可以根据需要调整
    margin_y = int((y_max - y_min) * expand_ratio)
    margin_x = int((x_max - x_min) * expand_ratio)
    
    y_min = max(0, y_min - margin_y)
    y_max = min(H_img, y_max + margin_y)
    x_min = max(0, x_min - margin_x)
    x_max = min(W_img, x_max + margin_x)
    
    bbox = (x_min, y_min, x_max, y_max)  # (x_min, y_min, x_max, y_max)
    
    print(f"  ✓ Expansion ratio: {expand_ratio*100:.0f}%")
    print(f"  ✓ Bounding box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
    print(f"  ✓ Crop size: {x_max - x_min} x {y_max - y_min}")

if bbox is not None:
    x_min, y_min, x_max, y_max = bbox
    
    # 裁剪RGB图像
    rgb_cropped = rgb_img[y_min:y_max, x_min:x_max].copy()
    
    # 裁剪深度图
    depth_cropped = depth_map[y_min:y_max, x_min:x_max].copy()
    
    # 裁剪mask
    mask_cropped = yolo_mask[y_min:y_max, x_min:x_max].copy()
    
    # 裁剪点云（保留裁剪区域内的所有点，包括外扩部分）
    if is_organized:
        # 有序点云：直接使用mask索引
        grid_points = points.reshape(H, W, 3)
        # 创建裁剪区域的mask（包含外扩部分）
        mask_crop_region = np.zeros((H, W), dtype=bool)
        mask_crop_region[y_min:y_max, x_min:x_max] = True
        # 保留裁剪区域内的所有点（包括mask外的点，这样才能看到外扩效果）
        points_cropped = grid_points[mask_crop_region]
        points_cropped = points_cropped.reshape(-1, 3)
        
        # 裁剪颜色（如果有）
        if colors is not None:
            grid_colors = colors.reshape(H, W, 3)
            colors_cropped = grid_colors[mask_crop_region]
            colors_cropped = colors_cropped.reshape(-1, 3)
        else:
            colors_cropped = None
    else:
        # 非有序点云：使用投影方法
        valid_mask = np.isfinite(points).all(axis=1)
        points_valid = points[valid_mask]
        
        if len(points_valid) > 0:
            min_xy = points_valid[:, :2].min(axis=0)
            max_xy = points_valid[:, :2].max(axis=0)
            scale = np.min(original_size / (max_xy - min_xy + 1e-6))
            offset = (original_size - (max_xy - min_xy) * scale) / 2
            
            xy = (points_valid[:, :2] - min_xy) * scale + offset
            u = np.round(xy[:, 0]).astype(int)
            v = np.round(xy[:, 1]).astype(int)
            
            valid_pixels = (u >= 0) & (u < original_size[0]) & (v >= 0) & (v < original_size[1])
            u = u[valid_pixels]
            v = v[valid_pixels]
            points_valid = points_valid[valid_pixels]
            
            # 检查点是否在裁剪区域内（包括外扩部分）
            in_crop_region = (u >= x_min) & (u < x_max) & (v >= y_min) & (v < y_max)
            # 保留裁剪区域内的所有点（包括mask外的点，这样才能看到外扩效果）
            points_cropped = points_valid[in_crop_region]
            
            if colors is not None:
                colors_valid = colors[valid_mask]
                colors_valid = colors_valid[valid_pixels]
                colors_cropped = colors_valid[in_crop_region]
            else:
                colors_cropped = None
        else:
            points_cropped = np.array([]).reshape(0, 3)
            colors_cropped = None
    
    print(f"  ✓ Cropped RGB: {rgb_cropped.shape}")
    print(f"  ✓ Cropped depth: {depth_cropped.shape}")
    print(f"  ✓ Cropped point cloud: {len(points_cropped)} points")
    
    # 创建裁剪后的点云对象（用于3D可视化）
    pcd_cropped = o3d.geometry.PointCloud()
    if len(points_cropped) > 0:
        pcd_cropped.points = o3d.utility.Vector3dVector(points_cropped)
        if colors_cropped is not None:
            pcd_cropped.colors = o3d.utility.Vector3dVector(colors_cropped)
        else:
            # 如果没有颜色，使用深度值作为颜色
            depths = points_cropped[:, 2]
            if len(depths) > 0:
                depth_min = depths.min()
                depth_max = depths.max()
                if depth_max > depth_min:
                    depths_norm = (depths - depth_min) / (depth_max - depth_min)
                    colors_depth = plt.cm.jet(depths_norm)[:, :3]
                    pcd_cropped.colors = o3d.utility.Vector3dVector(colors_depth)
else:
    rgb_cropped = None
    depth_cropped = None
    mask_cropped = None
    points_cropped = np.array([]).reshape(0, 3)
    pcd_cropped = None

# ========== 7. 可视化 ==========
print("\nGenerating visualizations...")

# 6.1 RGB图像
fig = plt.figure(figsize=(20, 12))

# RGB Image
ax1 = plt.subplot(2, 3, 1)
ax1.imshow(rgb_img_rgb)
ax1.set_title("RGB Image", fontsize=12, fontweight='bold')
ax1.axis('off')

# YOLO Mask
ax2 = plt.subplot(2, 3, 2)
ax2.imshow(yolo_mask, cmap='gray')
ax2.set_title("YOLO Mask (Fish Only)", fontsize=12, fontweight='bold')
ax2.axis('off')

# RGB + Mask Overlay
ax3 = plt.subplot(2, 3, 3)
rgb_with_mask = rgb_img_rgb.copy()
mask_contour = cv2.findContours(yolo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
if len(mask_contour) > 0:
    cv2.drawContours(rgb_with_mask, mask_contour, -1, (0, 255, 0), 3)
ax3.imshow(rgb_with_mask)
ax3.set_title("RGB + Mask Contour", fontsize=12, fontweight='bold')
ax3.axis('off')

# Depth Map (Pseudocolor)
ax4 = plt.subplot(2, 3, 4)
depth_vis = depth_map.copy()
valid_mask_depth = np.isfinite(depth_vis) & (depth_vis > 0)
if valid_mask_depth.sum() > 0:
    depth_min_vis = depth_vis[valid_mask_depth].min()
    depth_max_vis = depth_vis[valid_mask_depth].max()
    if depth_max_vis > depth_min_vis:
        depth_vis[valid_mask_depth] = (depth_vis[valid_mask_depth] - depth_min_vis) / (depth_max_vis - depth_min_vis)
    depth_vis = plt.cm.jet(depth_vis)[:, :, :3]
    depth_vis[~valid_mask_depth] = 0
else:
    depth_vis = np.zeros((*depth_map.shape, 3))
ax4.imshow(depth_vis)
ax4.set_title("Depth Map (Pseudocolor)", fontsize=12, fontweight='bold')
ax4.axis('off')

# Depth Map + Mask Overlay
ax5 = plt.subplot(2, 3, 5)
depth_with_mask = depth_vis.copy()
mask_overlay = yolo_mask > 0
depth_with_mask[mask_overlay] = depth_vis[mask_overlay] * 0.7 + np.array([0, 1, 0]) * 0.3  # Green overlay
ax5.imshow(depth_with_mask)
ax5.set_title("Depth Map + Mask", fontsize=12, fontweight='bold')
ax5.axis('off')

# Depth Map within Mask Region
ax6 = plt.subplot(2, 3, 6)
depth_masked = depth_map.copy()
depth_masked[yolo_mask == 0] = 0  # Only show mask region
depth_vis_masked = depth_masked.copy()
if valid_mask_depth.sum() > 0:
    depth_min_vis = depth_vis_masked[valid_mask_depth].min()
    depth_max_vis = depth_vis_masked[valid_mask_depth].max()
    if depth_max_vis > depth_min_vis:
        depth_vis_masked[valid_mask_depth] = (depth_vis_masked[valid_mask_depth] - depth_min_vis) / (depth_max_vis - depth_min_vis)
    depth_vis_masked = plt.cm.jet(depth_vis_masked)[:, :, :3]
    depth_vis_masked[~valid_mask_depth] = 0
else:
    depth_vis_masked = np.zeros((*depth_map.shape, 3))
ax6.imshow(depth_vis_masked)
ax6.set_title("Depth Map (Mask Region)", fontsize=12, fontweight='bold')
ax6.axis('off')

plt.tight_layout()
plt.savefig(sample_dir / "visualization_complete.png", dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved: {sample_dir / 'visualization_complete.png'}")
plt.show()

# ========== 7. Alignment Check ==========
print("\nChecking alignment between RGB, Mask, and Depth Map...")

# 计算mask和深度图的重叠区域
mask_binary = (yolo_mask > 0).astype(np.float32)
depth_binary = (np.isfinite(depth_map) & (depth_map > 0)).astype(np.float32)

# 重叠区域
overlap = mask_binary * depth_binary
overlap_ratio = overlap.sum() / (mask_binary.sum() + 1e-6)

print(f"  Mask area: {mask_binary.sum():.0f} pixels")
print(f"  Depth map valid area: {depth_binary.sum():.0f} pixels")
print(f"  Overlap area: {overlap.sum():.0f} pixels")
print(f"  Overlap ratio: {overlap_ratio*100:.1f}%")

# 可视化对齐检查
fig_align, axes_align = plt.subplots(1, 3, figsize=(15, 5))

# RGB with mask contour
axes_align[0].imshow(rgb_img_rgb)
mask_contour = cv2.findContours(yolo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
if len(mask_contour) > 0:
    cv2.drawContours(rgb_img_rgb, mask_contour, -1, (0, 255, 0), 3)
axes_align[0].imshow(rgb_img_rgb)
axes_align[0].set_title("RGB + Mask Contour", fontsize=12, fontweight='bold')
axes_align[0].axis('off')

# Depth map with mask contour
depth_vis_align = depth_map.copy()
valid_mask_align = np.isfinite(depth_vis_align) & (depth_vis_align > 0)
if valid_mask_align.sum() > 0:
    depth_min_align = depth_vis_align[valid_mask_align].min()
    depth_max_align = depth_vis_align[valid_mask_align].max()
    if depth_max_align > depth_min_align:
        depth_vis_align[valid_mask_align] = (depth_vis_align[valid_mask_align] - depth_min_align) / (depth_max_align - depth_min_align)
    depth_vis_align = plt.cm.jet(depth_vis_align)[:, :, :3]
    depth_vis_align[~valid_mask_align] = 0
else:
    depth_vis_align = np.zeros((*depth_map.shape, 3))

rgb_depth_align = depth_vis_align.copy()
mask_contour_depth = cv2.findContours(yolo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
if len(mask_contour_depth) > 0:
    # Convert to RGB for drawing
    rgb_depth_align_bgr = (rgb_depth_align * 255).astype(np.uint8)
    rgb_depth_align_bgr = cv2.cvtColor(rgb_depth_align_bgr, cv2.COLOR_RGB2BGR)
    cv2.drawContours(rgb_depth_align_bgr, mask_contour_depth, -1, (0, 255, 0), 3)
    rgb_depth_align = cv2.cvtColor(rgb_depth_align_bgr, cv2.COLOR_BGR2RGB) / 255.0
axes_align[1].imshow(rgb_depth_align)
axes_align[1].set_title("Depth Map + Mask Contour", fontsize=12, fontweight='bold')
axes_align[1].axis('off')

# Overlap visualization
overlap_vis = np.zeros((*depth_map.shape, 3))
overlap_vis[:, :, 0] = mask_binary  # Red for mask
overlap_vis[:, :, 2] = depth_binary  # Blue for depth
overlap_vis[:, :, 1] = overlap  # Green for overlap
axes_align[2].imshow(overlap_vis)
axes_align[2].set_title(f"Alignment Check (Overlap: {overlap_ratio*100:.1f}%)", fontsize=12, fontweight='bold')
axes_align[2].axis('off')

plt.tight_layout()
plt.savefig(sample_dir / "alignment_check.png", dpi=150, bbox_inches='tight')
print(f"✓ Alignment check saved: {sample_dir / 'alignment_check.png'}")
plt.show()

# ========== 8. 3D Point Cloud Visualization ==========
print("\nDisplaying 3D point clouds...")

# 可视化1：原始点云 vs Masked点云
print("  Visualization 1: Original vs Masked point cloud")
pcd_vis1 = pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色
pcd_masked_vis1 = pcd_masked.paint_uniform_color([1, 0, 0])  # 红色

vis1 = o3d.visualization.Visualizer()
vis1.create_window(window_name="Original vs Masked Point Cloud", width=1200, height=600)
vis1.add_geometry(pcd_vis1)
vis1.add_geometry(pcd_masked_vis1)

ctr1 = vis1.get_view_control()
ctr1.set_zoom(0.8)

vis1.run()
vis1.destroy_window()

# 可视化2：裁剪后的3D鱼（重点！）
if pcd_cropped is not None and len(points_cropped) > 0:
    print("  Visualization 2: Cropped 3D Fish (for dataset)")
    
    # 计算点云的中心和范围，用于设置合适的视角
    points_center = points_cropped.mean(axis=0)
    points_range = points_cropped.max(axis=0) - points_cropped.min(axis=0)
    max_range = points_range.max()
    
    # 创建可视化窗口（更大）
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name="Cropped 3D Fish (Dataset View)", width=1600, height=1200)
    vis2.add_geometry(pcd_cropped)
    
    # 设置视角，让鱼看起来更清晰
    ctr2 = vis2.get_view_control()
    
    # 计算合适的zoom值（让鱼填满窗口）
    # zoom值越小，物体越大
    zoom_factor = 0.3  # 更小的值 = 更大的显示
    
    # 设置相机位置，从侧面看鱼
    ctr2.set_lookat(points_center)
    ctr2.set_up([0, 0, 1])  # Z轴向上
    ctr2.set_front([1, 0, 0])  # 从X轴方向看
    ctr2.set_zoom(zoom_factor)
    
    # 设置渲染选项
    opt = vis2.get_render_option()
    opt.point_size = 3.0  # 更大的点
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # 深色背景，更突出
    
    vis2.run()
    vis2.destroy_window()
    
    print("  ✓ 3D fish visualization completed")
    print(f"    Point cloud center: ({points_center[0]:.2f}, {points_center[1]:.2f}, {points_center[2]:.2f})")
    print(f"    Point cloud range: {max_range:.2f}")
    print(f"    Total points: {len(points_cropped)}")
else:
    print("  ⚠️ No cropped point cloud to visualize")

# ========== 9. Save Results ==========
print("\nSaving processed results...")

# Resize到256x256
rgb_resized = cv2.resize(rgb_img, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
depth_map_resized = cv2.resize(depth_map, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
mask_resized = cv2.resize(yolo_mask, OUTPUT_SIZE, interpolation=cv2.INTER_NEAREST)

# 归一化深度图用于可视化
depth_normalized = depth_map_resized.copy()
valid_mask_depth = np.isfinite(depth_normalized) & (depth_normalized > 0)
if valid_mask_depth.sum() > 0:
    depth_min = depth_normalized[valid_mask_depth].min()
    depth_max = depth_normalized[valid_mask_depth].max()
    if depth_max > depth_min:
        depth_normalized[valid_mask_depth] = (depth_normalized[valid_mask_depth] - depth_min) / (depth_max - depth_min) * 255
    depth_normalized = depth_normalized.astype(np.uint8)
else:
    depth_normalized = np.zeros_like(depth_map_resized, dtype=np.uint8)

# 保存原始尺寸的文件
cv2.imwrite(str(sample_dir / "rgb_256.png"), rgb_resized)
cv2.imwrite(str(sample_dir / "depth_256.png"), depth_normalized)
cv2.imwrite(str(sample_dir / "mask_256.png"), mask_resized)
np.save(str(sample_dir / "depth_256.npy"), depth_map_resized)

print(f"✓ RGB saved: {sample_dir / 'rgb_256.png'}")
print(f"✓ Depth map saved: {sample_dir / 'depth_256.png'}")
print(f"✓ Mask saved: {sample_dir / 'mask_256.png'}")
print(f"✓ Raw depth values saved: {sample_dir / 'depth_256.npy'}")

# 保存裁剪后的文件（用于数据集）
if bbox is not None and rgb_cropped is not None:
    # ========== 9.1. 可视化裁剪后的RGB + 深度图 + Mask叠加 ==========
    print("\nGenerating RGB + Depth + Mask overlay visualization (cropped, no background)...")
    
    # 准备深度图伪彩色
    depth_vis_crop = depth_cropped.copy()
    valid_mask_crop_vis = np.isfinite(depth_vis_crop) & (depth_vis_crop > 0)
    if valid_mask_crop_vis.sum() > 0:
        depth_min_crop_vis = depth_vis_crop[valid_mask_crop_vis].min()
        depth_max_crop_vis = depth_vis_crop[valid_mask_crop_vis].max()
        if depth_max_crop_vis > depth_min_crop_vis:
            depth_vis_crop[valid_mask_crop_vis] = (depth_vis_crop[valid_mask_crop_vis] - depth_min_crop_vis) / (depth_max_crop_vis - depth_min_crop_vis)
        depth_colormap_crop = plt.cm.jet(depth_vis_crop)[:, :, :3]
        depth_colormap_crop[~valid_mask_crop_vis] = 0
    else:
        depth_colormap_crop = np.zeros((*depth_cropped.shape, 3))
    
    # RGB图像（转换为RGB格式）
    rgb_crop_rgb = cv2.cvtColor(rgb_cropped, cv2.COLOR_BGR2RGB)
    mask_binary_crop = mask_cropped > 0
    
    fig_overlay = plt.figure(figsize=(18, 6))
    
    # 方法1：RGB + 深度图伪彩色叠加（只在mask区域）
    ax1 = plt.subplot(1, 3, 1)
    overlay1 = rgb_crop_rgb.copy().astype(np.float32) / 255.0
    # 只在mask区域叠加深度图
    overlay1[mask_binary_crop] = overlay1[mask_binary_crop] * 0.5 + depth_colormap_crop[mask_binary_crop] * 0.5
    ax1.imshow(overlay1)
    ax1.set_title("RGB + Depth (Mask Region Only)\n(No Background Interference)", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 方法2：深度图伪彩色 + RGB叠加（整体，但mask区域更突出）
    ax2 = plt.subplot(1, 3, 2)
    overlay2 = rgb_crop_rgb.copy().astype(np.float32) / 255.0 * 0.4 + depth_colormap_crop * 0.6
    # 在mask区域增强深度信息
    overlay2[mask_binary_crop] = rgb_crop_rgb[mask_binary_crop].astype(np.float32) / 255.0 * 0.3 + depth_colormap_crop[mask_binary_crop] * 0.7
    ax2.imshow(overlay2)
    ax2.set_title("RGB + Depth (Enhanced in Mask)\n(Depth Information Highlighted)", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 方法3：纯深度图伪彩色 + Mask轮廓
    ax3 = plt.subplot(1, 3, 3)
    overlay3 = depth_colormap_crop.copy()
    # 在mask轮廓处画绿色线
    mask_contour_crop = cv2.findContours(mask_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(mask_contour_crop) > 0:
        # 转换为BGR格式用于drawContours
        overlay3_bgr = (overlay3 * 255).astype(np.uint8)
        overlay3_bgr = cv2.cvtColor(overlay3_bgr, cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay3_bgr, mask_contour_crop, -1, (0, 255, 0), 3)
        overlay3 = cv2.cvtColor(overlay3_bgr, cv2.COLOR_BGR2RGB) / 255.0
    ax3.imshow(overlay3)
    ax3.set_title("Depth Map + Mask Contour\n(Pure Depth Visualization)", fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(sample_dir / "visualization_cropped_overlay.png", dpi=150, bbox_inches='tight')
    print(f"  ✓ Overlay visualization saved: {sample_dir / 'visualization_cropped_overlay.png'}")
    plt.show()
    
    # Resize裁剪后的图像到256x256
    rgb_cropped_resized = cv2.resize(rgb_cropped, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    depth_cropped_resized = cv2.resize(depth_cropped, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    mask_cropped_resized = cv2.resize(mask_cropped, OUTPUT_SIZE, interpolation=cv2.INTER_NEAREST)
    
    # 归一化裁剪后的深度图
    depth_cropped_normalized = depth_cropped_resized.copy()
    valid_mask_crop = np.isfinite(depth_cropped_normalized) & (depth_cropped_normalized > 0)
    if valid_mask_crop.sum() > 0:
        depth_min_crop = depth_cropped_normalized[valid_mask_crop].min()
        depth_max_crop = depth_cropped_normalized[valid_mask_crop].max()
        if depth_max_crop > depth_min_crop:
            depth_cropped_normalized[valid_mask_crop] = (depth_cropped_normalized[valid_mask_crop] - depth_min_crop) / (depth_max_crop - depth_min_crop) * 255
        depth_cropped_normalized = depth_cropped_normalized.astype(np.uint8)
    else:
        depth_cropped_normalized = np.zeros_like(depth_cropped_resized, dtype=np.uint8)
    
    # 保存裁剪后的文件
    cv2.imwrite(str(sample_dir / "rgb_cropped_256.png"), rgb_cropped_resized)
    cv2.imwrite(str(sample_dir / "depth_cropped_256.png"), depth_cropped_normalized)
    cv2.imwrite(str(sample_dir / "mask_cropped_256.png"), mask_cropped_resized)
    np.save(str(sample_dir / "depth_cropped_256.npy"), depth_cropped_resized)
    
    # 保存裁剪后的点云
    if len(points_cropped) > 0:
        o3d.io.write_point_cloud(str(sample_dir / "pointcloud_cropped.ply"), pcd_cropped)
        print(f"✓ Cropped point cloud saved: {sample_dir / 'pointcloud_cropped.ply'}")
    
    # 保存裁剪信息（边界框）
    # 确保所有值都是Python原生类型，不是numpy类型
    crop_info = {
        'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
        'original_size': [int(original_size[0]), int(original_size[1])],
        'cropped_size': [int(rgb_cropped.shape[1]), int(rgb_cropped.shape[0])],
        'resized_size': [int(OUTPUT_SIZE[0]), int(OUTPUT_SIZE[1])]
    }
    import json
    with open(sample_dir / "crop_info.json", 'w') as f:
        json.dump(crop_info, f, indent=2)
    
    print(f"✓ Cropped RGB saved: {sample_dir / 'rgb_cropped_256.png'}")
    print(f"✓ Cropped depth map saved: {sample_dir / 'depth_cropped_256.png'}")
    print(f"✓ Cropped mask saved: {sample_dir / 'mask_cropped_256.png'}")
    print(f"✓ Cropped raw depth values saved: {sample_dir / 'depth_cropped_256.npy'}")
    print(f"✓ Crop info saved: {sample_dir / 'crop_info.json'}")

# ========== 10. Statistics ==========
print("\n" + "=" * 60)
print("Processing Statistics")
print("=" * 60)
print(f"RGB图像尺寸: {rgb_img.shape}")
print(f"YOLO Mask前景比例: {(yolo_mask > 0).sum() / yolo_mask.size * 100:.1f}%")
print(f"点云总数: {len(points)}")
print(f"Mask内点云数: {len(points_masked)} ({len(points_masked)/len(points)*100:.1f}%)")
print(f"深度图有效像素: {valid_mask.sum()} ({valid_mask.sum()/depth_map.size*100:.1f}%)")
print(f"深度范围: [{depth_min:.3f}, {depth_max:.3f}]")
print(f"输出尺寸: {OUTPUT_SIZE}")

print("\n✓ 单个样本测试完成！")

