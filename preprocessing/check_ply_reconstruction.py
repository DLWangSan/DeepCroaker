"""
Check if A-class data (depth map + RGB + mask) can reconstruct PLY point cloud
"""
import numpy as np
import cv2
from pathlib import Path
import json
import open3d as o3d
import matplotlib.pyplot as plt

CONFIG_FILE = Path("config.json")
with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    config = json.load(f)

ORGANIZED_DIR = Path(config['paths']['organized_dir'])
OUTPUT_SIZE = tuple(config['processing']['output_size'])

# 选择测试样本
sample_dir = ORGANIZED_DIR / "wild" / "sample_0001"

print("=" * 60)
print("检查PLY点云重建能力")
print("=" * 60)
print(f"测试样本: {sample_dir}")

# 1. 加载原始PLY点云
ply_file = sample_dir / "pointcloud.ply"
if not ply_file.exists():
    print(f"错误：未找到PLY文件 {ply_file}")
    exit(1)

pcd_original = o3d.io.read_point_cloud(str(ply_file))
points_original = np.asarray(pcd_original.points)
colors_original = np.asarray(pcd_original.colors) if pcd_original.has_colors() else None
print(f"✓ 原始点云: {len(points_original)} 个点")
if colors_original is not None:
    print(f"✓ 原始点云包含颜色信息")

# 2. 加载处理后的256×256数据（直接使用，无需回到原始尺寸）
rgb_256_file = sample_dir / "rgb_256.png"
depth_256_file = sample_dir / "depth_256.npy"
mask_file = sample_dir / "mask_256.png"

if not (rgb_256_file.exists() and depth_256_file.exists() and mask_file.exists()):
    print(f"Error: processed files not found")
    print(f"  Please run: python 3_generate_depth_maps_with_yolo.py")
    exit(1)

print("Using batch-processed data (extreme values removed)")
rgb_img = cv2.imread(str(rgb_256_file))
depth_map = np.load(str(depth_256_file))
mask_img = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

print(f"  RGB: {rgb_img.shape}, Depth: {depth_map.shape}, Mask: {mask_img.shape}")

mask_binary_check = mask_img > 0
valid_in_mask = np.isfinite(depth_map) & (depth_map > 0) & mask_binary_check

if valid_in_mask.sum() > 0:
    depth_in_mask = depth_map[valid_in_mask]
    print(f"  Depth values in mask: min={depth_in_mask.min():.4f}, max={depth_in_mask.max():.4f}, range={depth_in_mask.max()-depth_in_mask.min():.4f}")
    print(f"  Depth map has been cleaned (5%-95% percentile) during data processing")
    
    if depth_in_mask.max() - depth_in_mask.min() > 100:
        print(f"  Warning: depth value range is still large, extreme values may not be properly removed")
        print(f"      Please check if 3_generate_depth_maps_with_yolo.py correctly removed extreme values")
else:
    print("  Warning: no valid depth values in mask region")

print(f"RGB image: {rgb_img.shape}")
print(f"Depth map: {depth_map.shape}")
print(f"Mask: {mask_img.shape}")

print("\nReconstructing point cloud from 256x256 depth map...")

mask_binary = mask_img > 0
valid_pixels = np.where(mask_binary)

if len(valid_pixels[0]) == 0:
    print("Error: mask is empty")
    exit(1)

H, W = OUTPUT_SIZE[1], OUTPUT_SIZE[0]

print(f"  Reconstructing point cloud using 256x256 data")
print(f"  Image size: {W}x{H}")

reconstructed_points = []
reconstructed_colors = []

for i, j in zip(valid_pixels[0], valid_pixels[1]):
    z = depth_map[i, j]
    
    if np.isfinite(z) and z > 0:
        x = j
        y = i
        
        if rgb_img is not None and 0 <= i < rgb_img.shape[0] and 0 <= j < rgb_img.shape[1]:
            b, g, r = rgb_img[i, j]
            color_rgb = [r / 255.0, g / 255.0, b / 255.0]
            color_rgb = [max(0.0, min(1.0, c)) for c in color_rgb]
            
            reconstructed_points.append([x, y, z])
            reconstructed_colors.append(color_rgb)
        else:
            continue

reconstructed_points = np.array(reconstructed_points)
reconstructed_colors = np.array(reconstructed_colors) if len(reconstructed_colors) > 0 else None

print(f"Reconstructed point cloud: {len(reconstructed_points)} points")

print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)

print(f"Original point cloud:")
print(f"  Points: {len(points_original)}")
print(f"  Range: X[{points_original[:, 0].min():.2f}, {points_original[:, 0].max():.2f}]")
print(f"         Y[{points_original[:, 1].min():.2f}, {points_original[:, 1].max():.2f}]")
print(f"         Z[{points_original[:, 2].min():.2f}, {points_original[:, 2].max():.2f}]")

print(f"\nReconstructed point cloud (256x256):")
print(f"  Points: {len(reconstructed_points)}")
if len(reconstructed_points) > 0:
    print(f"  Range: X[{reconstructed_points[:, 0].min():.2f}, {reconstructed_points[:, 0].max():.2f}]")
    print(f"         Y[{reconstructed_points[:, 1].min():.2f}, {reconstructed_points[:, 1].max():.2f}]")
    print(f"         Z[{reconstructed_points[:, 2].min():.4f}, {reconstructed_points[:, 2].max():.4f}]")

print(f"\nConclusion:")
print(f"  - Using 256x256 depth map and RGB to reconstruct point cloud")
print(f"  - X, Y coordinates use pixel coordinates (0-255), Z uses depth values")
print(f"  - Reconstructed point cloud maintains cropped and resized structure")
print(f"  - Reconstructed points: {len(reconstructed_points)} (mask region)")

print("\n" + "=" * 60)
print("2D Visualization: Depth Map Overlay")
print("=" * 60)

depth_vis = depth_map.copy()
mask_binary_vis = mask_img > 0

valid_mask_depth = np.isfinite(depth_vis) & (depth_vis > 0) & mask_binary_vis

if valid_mask_depth.sum() > 0:
    depth_min = depth_vis[valid_mask_depth].min()
    depth_max = depth_vis[valid_mask_depth].max()
    
    print(f"  Depth value range (mask region, extreme values removed): [{depth_min:.4f}, {depth_max:.4f}]")
    print(f"  Depth value variation: {depth_max - depth_min:.4f}")
    
    if depth_max > depth_min:
        valid_all = np.isfinite(depth_vis) & (depth_vis > 0)
        depth_vis[valid_all] = (depth_vis[valid_all] - depth_min) / (depth_max - depth_min)
        depth_vis = np.clip(depth_vis, 0, 1)
        depth_colormap = plt.cm.jet(depth_vis)[:, :, :3]
    else:
        depth_colormap = np.zeros((*depth_map.shape, 3))
        depth_colormap[valid_mask_depth] = [0, 0, 1]
    
    depth_colormap[~mask_binary_vis] = 0
else:
    print("  Warning: no valid depth values in mask region")
    depth_colormap = np.zeros((*depth_map.shape, 3))

rgb_img_rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
mask_binary_vis = mask_img > 0

fig = plt.figure(figsize=(20, 12))

ax1 = plt.subplot(2, 3, 1)
ax1.imshow(rgb_img_rgb)
ax1.set_title("RGB Image", fontsize=12, fontweight='bold')
ax1.axis('off')

ax2 = plt.subplot(2, 3, 2)
ax2.imshow(depth_colormap)
ax2.set_title("Depth Map (Pseudocolor)", fontsize=12, fontweight='bold')
ax2.axis('off')

ax3 = plt.subplot(2, 3, 3)
ax3.imshow(mask_img, cmap='gray')
ax3.set_title("Mask", fontsize=12, fontweight='bold')
ax3.axis('off')

ax4 = plt.subplot(2, 3, 4)
overlay1 = rgb_img_rgb.copy().astype(np.float32) / 255.0 * 0.4 + depth_colormap * 0.6
ax4.imshow(overlay1)
ax4.set_title("RGB + Depth (Overall)", fontsize=12, fontweight='bold')
ax4.axis('off')

ax5 = plt.subplot(2, 3, 5)
overlay2 = rgb_img_rgb.copy().astype(np.float32) / 255.0
overlay2[mask_binary_vis] = depth_colormap[mask_binary_vis]
ax5.imshow(overlay2)
ax5.set_title("RGB + Depth (Only Mask Region)", fontsize=12, fontweight='bold')
ax5.axis('off')

ax6 = plt.subplot(2, 3, 6)
overlay3 = depth_colormap.copy()
mask_contour = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
if len(mask_contour) > 0:
    overlay3_bgr = (overlay3 * 255).astype(np.uint8)
    overlay3_bgr = cv2.cvtColor(overlay3_bgr, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay3_bgr, mask_contour, -1, (0, 255, 0), 2)
    overlay3 = cv2.cvtColor(overlay3_bgr, cv2.COLOR_BGR2RGB) / 255.0
ax6.imshow(overlay3)
ax6.set_title("Depth Map + Mask Contour", fontsize=12, fontweight='bold')
ax6.axis('off')

plt.tight_layout()
plt.show()

print("2D visualization completed")

if len(reconstructed_points) > 0:
    reconstructed_points = np.array(reconstructed_points)
    reconstructed_colors = np.array(reconstructed_colors) if len(reconstructed_colors) > 0 else None
    
    if reconstructed_colors is not None:
        if len(reconstructed_colors) != len(reconstructed_points):
            print(f"  Warning: point count ({len(reconstructed_points)}) and color count ({len(reconstructed_colors)}) mismatch")
            min_len = min(len(reconstructed_points), len(reconstructed_colors))
            reconstructed_points = reconstructed_points[:min_len]
            reconstructed_colors = reconstructed_colors[:min_len]
    
    pcd_reconstructed = o3d.geometry.PointCloud()
    pcd_reconstructed.points = o3d.utility.Vector3dVector(reconstructed_points)
    
    if reconstructed_colors is not None and len(reconstructed_colors) == len(reconstructed_points):
        reconstructed_colors = np.clip(reconstructed_colors, 0.0, 1.0)
        pcd_reconstructed.colors = o3d.utility.Vector3dVector(reconstructed_colors)
        print(f"  Point cloud contains RGB color information")
    else:
        print(f"  Using depth values as colors (pseudocolor)")
        depths = reconstructed_points[:, 2]
        if len(depths) > 0:
            depth_min = depths.min()
            depth_max = depths.max()
            if depth_max > depth_min:
                depths_norm = (depths - depth_min) / (depth_max - depth_min)
                colors_depth = plt.cm.jet(depths_norm)[:, :3]
                pcd_reconstructed.colors = o3d.utility.Vector3dVector(colors_depth)
    
    output_file = sample_dir / "pointcloud_reconstructed.ply"
    o3d.io.write_point_cloud(str(output_file), pcd_reconstructed)
    print(f"\nReconstructed point cloud saved: {output_file}")
    
    print("\n" + "=" * 60)
    print("3D Visualization")
    print("=" * 60)
    
    print("Visualization 1: Reconstructed point cloud (256x256 with RGB)")
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name="Reconstructed Point Cloud (256x256 with RGB)", width=1600, height=1200)
    vis1.add_geometry(pcd_reconstructed)
    
    ctr1 = vis1.get_view_control()
    points_center = reconstructed_points.mean(axis=0)
    ctr1.set_lookat(points_center)
    ctr1.set_up([0, 0, 1])
    ctr1.set_front([1, 0, 0])
    ctr1.set_zoom(0.5)
    
    opt1 = vis1.get_render_option()
    opt1.point_size = 3.0
    opt1.show_coordinate_frame = True
    opt1.background_color = np.asarray([0.1, 0.1, 0.1])
    
    vis1.run()
    vis1.destroy_window()
    
    print("Visualization 2: Original vs Reconstructed point cloud (comparison)")
    pcd_original_vis = pcd_original.paint_uniform_color([0.5, 0.5, 0.5])
    
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name="Original vs Reconstructed Point Cloud", width=1600, height=800)
    vis2.add_geometry(pcd_original_vis)
    vis2.add_geometry(pcd_reconstructed)
    
    ctr2 = vis2.get_view_control()
    ctr2.set_zoom(0.7)
    
    vis2.run()
    vis2.destroy_window()
    
    print("\n3D visualization completed")
else:
    print("\nCannot reconstruct point cloud (mask is empty or depth map is invalid)")

print("\n" + "=" * 60)
print("Check completed")
print("=" * 60)

