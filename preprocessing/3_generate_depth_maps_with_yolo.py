"""
Generate depth maps from PLY point clouds and masks using YOLO segmentation model
"""
import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm
import open3d as o3d

CONFIG_FILE = Path(__file__).parent / "config.json"

def load_config():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"配置文件不存在: {CONFIG_FILE}")
    
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

config = load_config()
ORGANIZED_DIR = Path(config['paths']['organized_dir'])
OUTPUT_SIZE = tuple(config['processing']['output_size'])

YOLO_MODEL_PATH = config.get('yolo', {}).get('model_path', r"E:\ECSF\dahuangyu\code\runs\segment\train4\weights\best.pt")
YOLO_CONF_THRES = config.get('yolo', {}).get('conf_thres', 0.5)

def load_yolo_model():
    try:
        from ultralytics import YOLO
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading YOLO model: {YOLO_MODEL_PATH}")
        print(f"Device: {device.upper()}")
        if device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("  Warning: Using CPU, processing will be slow")
        
        model = YOLO(YOLO_MODEL_PATH)
        print(f"YOLO model loaded successfully")
        return model
    except ImportError:
        raise ImportError("Please install ultralytics: pip install ultralytics")
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model: {e}")

def get_yolo_mask(rgb_image, yolo_model, conf_thres=0.5):
    """Generate mask using YOLO model"""
    results = yolo_model.predict(rgb_image, conf=conf_thres, verbose=False)
    result = results[0]
    
    if result.masks is not None and len(result.masks) > 0:
        boxes = result.boxes
        if len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confs))
            
            mask = result.masks[best_idx].data.cpu().numpy()
            if mask.ndim == 3:
                mask = mask[0]
            
            mask = (mask * 255).astype(np.uint8)
            
            if mask.shape != rgb_image.shape[:2]:
                mask = cv2.resize(mask, (rgb_image.shape[1], rgb_image.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            return mask
    
    return np.zeros(rgb_image.shape[:2], dtype=np.uint8)

def load_ply_pointcloud(ply_file):
    """Load PLY point cloud"""
    pcd = o3d.io.read_point_cloud(str(ply_file))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return points, colors

def extract_depth_from_organized_pointcloud(points, image_size):
    """Extract depth map from organized point cloud (Zivid camera output)"""
    W, H = image_size
    expected_points = H * W
    actual_points = len(points)
    
    if actual_points == expected_points:
        grid_points = points.reshape(H, W, 3)
        depth_map = grid_points[:, :, 2].copy()
        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
        return depth_map, True
    else:
        return None, False

def process_sample(sample_dir, yolo_model, output_size=None):
    """Process single sample: generate depth map and mask using YOLO"""
    if output_size is None:
        output_size = OUTPUT_SIZE
    
    sample_dir = Path(sample_dir)
    
    ply_file = sample_dir / "pointcloud.ply"
    if not ply_file.exists():
        print(f"Warning: PLY file not found in {sample_dir}")
        return None
    
    rgb_file = sample_dir / "rgb.png"
    if not rgb_file.exists():
        print(f"Warning: RGB image not found in {sample_dir}")
        return None
    
    rgb_img = cv2.imread(str(rgb_file))
    if rgb_img is None:
        print(f"Warning: failed to load RGB image {rgb_file}")
        return None
    
    original_size = (rgb_img.shape[1], rgb_img.shape[0])
    
    try:
        yolo_mask = get_yolo_mask(rgb_img, yolo_model, conf_thres=YOLO_CONF_THRES)
    except Exception as e:
        print(f"Warning: YOLO mask generation failed {sample_dir}: {e}")
        return None
    
    try:
        points, colors = load_ply_pointcloud(ply_file)
        depth_map, is_organized = extract_depth_from_organized_pointcloud(points, original_size)
        
        if depth_map is None:
            print(f"Warning: failed to generate depth map {sample_dir}")
            return None
        
        if yolo_mask.shape != depth_map.shape:
            yolo_mask = cv2.resize(yolo_mask, (depth_map.shape[1], depth_map.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
        
        final_mask = yolo_mask.copy()
        
        mask_binary = yolo_mask > 0
        if mask_binary.sum() > 0:
            coords = np.column_stack(np.where(mask_binary))
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            H_img, W_img = rgb_img.shape[:2]
            expand_ratio = 0.1
            margin_y = int((y_max - y_min) * expand_ratio)
            margin_x = int((x_max - x_min) * expand_ratio)
            
            y_min = max(0, y_min - margin_y)
            y_max = min(H_img, y_max + margin_y)
            x_min = max(0, x_min - margin_x)
            x_max = min(W_img, x_max + margin_x)
            
            rgb_cropped = rgb_img[y_min:y_max, x_min:x_max].copy()
            depth_cropped = depth_map[y_min:y_max, x_min:x_max].copy()
            mask_cropped = yolo_mask[y_min:y_max, x_min:x_max].copy()
        else:
            rgb_cropped = rgb_img
            depth_cropped = depth_map
            mask_cropped = yolo_mask
        
        depth_map_resized = cv2.resize(depth_cropped, output_size, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_cropped, output_size, interpolation=cv2.INTER_NEAREST)
        rgb_resized = cv2.resize(rgb_cropped, output_size, interpolation=cv2.INTER_LINEAR)
        
    except Exception as e:
        print(f"Warning: failed to process PLY file {ply_file}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    mask_binary_for_clip = mask_resized > 0
    valid_mask_for_clip = np.isfinite(depth_map_resized) & (depth_map_resized > 0) & mask_binary_for_clip
    
    if valid_mask_for_clip.sum() > 0:
        depth_values_in_mask = depth_map_resized[valid_mask_for_clip]
        depth_min_percentile = np.percentile(depth_values_in_mask, 5)
        depth_max_percentile = np.percentile(depth_values_in_mask, 95)
        
        depth_map_cleaned = np.clip(depth_map_resized, depth_min_percentile, depth_max_percentile)
        depth_map_cleaned[~np.isfinite(depth_map_resized)] = 0
        depth_map_cleaned[depth_map_resized <= 0] = 0
    else:
        depth_map_cleaned = depth_map_resized.copy()
    
    depth_normalized = depth_map_cleaned.copy()
    valid_mask = np.isfinite(depth_normalized) & (depth_normalized > 0) & mask_binary_for_clip
    if valid_mask.sum() > 0:
        depth_min_vis = depth_normalized[valid_mask].min()
        depth_max_vis = depth_normalized[valid_mask].max()
        if depth_max_vis > depth_min_vis:
            depth_normalized[valid_mask] = (depth_normalized[valid_mask] - depth_min_vis) / (depth_max_vis - depth_min_vis) * 255
        depth_normalized = depth_normalized.astype(np.uint8)
    else:
        depth_normalized = np.zeros_like(depth_map_cleaned, dtype=np.uint8)
    
    output_dir = sample_dir
    cv2.imwrite(str(output_dir / "rgb_256.png"), rgb_resized)
    cv2.imwrite(str(output_dir / "depth_256.png"), depth_normalized)
    cv2.imwrite(str(output_dir / "mask_256.png"), mask_resized)
    np.save(str(output_dir / "depth_256.npy"), depth_map_cleaned)
    
    return {
        'rgb': str(output_dir / "rgb_256.png"),
        'depth': str(output_dir / "depth_256.png"),
        'depth_raw': str(output_dir / "depth_256.npy"),
        'mask': str(output_dir / "mask_256.png")
    }

def main():
    organized_dir = ORGANIZED_DIR
    index_file = organized_dir / "data_index.json"
    
    if not index_file.exists():
        print("Error: please run 1_scan_and_organize_data.py first")
        return
    
    print("=" * 60)
    print("Loading YOLO segmentation model")
    print("=" * 60)
    try:
        yolo_model = load_yolo_model()
    except Exception as e:
        print(f"Error: failed to load YOLO model: {e}")
        return
    
    with open(index_file, 'r', encoding='utf-8') as f:
        data_index = json.load(f)
    
    print("\n" + "=" * 60)
    print("Generate depth maps from PLY point clouds, generate masks using YOLO")
    print("=" * 60)
    
    all_samples = []
    for label in ['wild', 'farmed']:
        all_samples.extend(data_index[label])
    
    print(f"Total {len(all_samples)} samples to process\n")
    
    results = []
    failed_samples = []
    
    for sample_info in tqdm(all_samples, desc="Processing samples"):
        sample_id = sample_info['sample_id']
        label = sample_info['label']
        sample_dir = organized_dir / label / sample_id
        
        result = process_sample(sample_dir, yolo_model)
        if result is not None:
            result['sample_id'] = sample_id
            result['label'] = label
            results.append(result)
        else:
            failed_samples.append(sample_id)
    
    for label in ['wild', 'farmed']:
        for sample_info in data_index[label]:
            sample_id = sample_info['sample_id']
            sample_dir = organized_dir / label / sample_id
            
            if (sample_dir / "depth_256.png").exists():
                sample_info['files']['rgb_256'] = f"{label}/{sample_id}/rgb_256.png"
                sample_info['files']['depth_256'] = f"{label}/{sample_id}/depth_256.png"
                sample_info['files']['depth_256_raw'] = f"{label}/{sample_id}/depth_256.npy"
                sample_info['files']['mask_256'] = f"{label}/{sample_id}/mask_256.png"
    
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(data_index, f, indent=2, ensure_ascii=False)
    
    print(f"\nDone! Successfully processed {len(results)} samples")
    if failed_samples:
        print(f"Failed samples: {len(failed_samples)}")
        print(f"Failed sample IDs: {failed_samples[:10]}...")

if __name__ == "__main__":
    main()

