"""
Process B-class data (action camera data): generate masks using YOLO, crop based on mask bounding box, and resize to 256x256.

Processing steps:
1. Generate mask (fish body only) using YOLO segmentation model.
2. Calculate bounding box from mask (with 10% expansion margin).
3. Crop RGB and mask based on bounding box.
4. Resize to 256x256 and save.

Output files:
- rgb.png: RGB image (256x256)
- mask.png: Mask (256x256, 0/255)

Note: B-class data does not have depth information.
"""
import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm
from ultralytics import YOLO
import torch

CONFIG_FILE = Path(__file__).parent / "config.json"

def load_config():
    """Loads configuration from config.json."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

config = load_config()
B_CLASS_SOURCE_DIR = Path(r"E:\ECSF\dahuangyu\code\datasets")
DATASET_DIR = Path(config['paths']['dataset_dir'])
OUTPUT_SIZE = tuple(config['processing']['output_size'])
YOLO_MODEL_PATH = config.get('yolo', {}).get('model_path', r"E:\ECSF\dahuangyu\code\runs\segment\train4\weights\best.pt")
YOLO_CONF_THRES = config.get('yolo', {}).get('conf_thres', 0.5)
EXPAND_RATIO = 0.1  # 10% expansion margin for bounding box

def load_yolo_model():
    """Loads the YOLO segmentation model."""
    try:
        from ultralytics import YOLO
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading YOLO model: {YOLO_MODEL_PATH}")
        print(f"Device: {device.upper()}")
        if device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
        else:
            print("  Warning: Using CPU, processing will be slower.")
        
        model = YOLO(YOLO_MODEL_PATH)
        print(f"✓ YOLO model loaded successfully.")
        return model
    except ImportError:
        raise ImportError("ultralytics not installed: pip install ultralytics")
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model: {e}")

def get_yolo_mask(rgb_image, yolo_model, conf_thres=0.5):
    """
    Generates a mask on an RGB image using the YOLO model.
    
    Args:
        rgb_image: RGB image (H, W, 3) in BGR format (OpenCV format).
        yolo_model: YOLO model.
        conf_thres: Confidence threshold.
    
    Returns:
        mask: Binary mask (H, W) uint8, 255=foreground (fish), 0=background.
    """
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

def process_b_class_image(img_path, yolo_model, output_dir, sample_id):
    """
    Processes a single B-class image.
    
    Processing steps:
    1. Generate mask using YOLO.
    2. Calculate bounding box from mask (with expansion margin).
    3. Crop RGB and mask based on bounding box.
    4. Resize to 256x256 and save.
    
    Args:
        img_path: Image path.
        yolo_model: YOLO model.
        output_dir: Output directory.
        sample_id: Sample ID.
    
    Returns:
        dict: Processing result information.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    original_size = img.shape[:2]
    
    # Generate mask using YOLO
    mask = get_yolo_mask(img, yolo_model, conf_thres=YOLO_CONF_THRES)
    
    # Calculate bounding box from mask (with expansion margin)
    mask_binary = mask > 0
    if mask_binary.sum() > 0:
        coords = np.column_stack(np.where(mask_binary))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        H_img, W_img = img.shape[:2]
        margin_y = int((y_max - y_min) * EXPAND_RATIO)
        margin_x = int((x_max - x_min) * EXPAND_RATIO)
        
        y_min = max(0, y_min - margin_y)
        y_max = min(H_img, y_max + margin_y)
        x_min = max(0, x_min - margin_x)
        x_max = min(W_img, x_max + margin_x)
        
        # Crop RGB and mask based on bounding box
        img_cropped = img[y_min:y_max, x_min:x_max].copy()
        mask_cropped = mask[y_min:y_max, x_min:x_max].copy()
    else:
        # If mask is empty, use full image
        img_cropped = img
        mask_cropped = mask
    
    # Resize to output size
    img_resized = cv2.resize(img_cropped, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_cropped, OUTPUT_SIZE, interpolation=cv2.INTER_NEAREST)
    
    # Save files
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "rgb.png"), img_resized)
    cv2.imwrite(str(output_dir / "mask.png"), mask_resized)
    
    return {
        'sample_id': sample_id,
        'original_size': original_size,
        'resized_size': OUTPUT_SIZE,
        'mask_foreground_ratio': (mask_resized > 0).sum() / mask_resized.size * 100
    }

def main():
    """Main function to process B-class data."""
    print("=" * 60)
    print("Processing B-class Data (Action Camera Data)")
    print("=" * 60)
    
    if not B_CLASS_SOURCE_DIR.exists():
        print(f"Error: B-class data directory not found: {B_CLASS_SOURCE_DIR}")
        return
    
    print("\nLoading YOLO model...")
    try:
        yolo_model = load_yolo_model()
    except Exception as e:
        print(f"Error: Could not load YOLO model: {e}")
        return
    
    b_class_dir = DATASET_DIR / "B_class"
    b_class_dir.mkdir(parents=True, exist_ok=True)
    
    splits = ['train', 'valid', 'test']
    
    all_samples = []
    total_processed = 0
    sample_counter = {'wild': 0, 'farmed': 0}
    
    for split in splits:
        images_dir = B_CLASS_SOURCE_DIR / split / "images"
        labels_dir = B_CLASS_SOURCE_DIR / split / "labels"
        
        if not images_dir.exists():
            print(f"\nInfo: {split}/images directory not found, skipping.")
            continue
        
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(images_dir.glob(f"*{ext}"))
        
        if not image_files:
            print(f"\nInfo: No image files found in {split}, skipping.")
            continue
        
        print(f"\nProcessing {split} set: Found {len(image_files)} image files.")
        
        for img_file in tqdm(image_files, desc=f"Processing {split} data"):
            label = None
            filename_lower = img_file.name.lower()
            
            if 'wild' in filename_lower or '野生' in filename_lower:
                label = 'wild'
            elif 'farmed' in filename_lower or '养殖' in filename_lower:
                label = 'farmed'
            else:
                if labels_dir.exists():
                    label_file = labels_dir / (img_file.stem + '.txt')
                    if label_file.exists():
                        try:
                            with open(label_file, 'r') as f:
                                first_line = f.readline().strip()
                                if first_line:
                                    class_id = int(first_line.split()[0])
                                    label = 'wild' if class_id == 0 else 'farmed'
                        except Exception as e:
                            pass
            
            if label is None:
                print(f"Warning: Could not determine label for {img_file.name}, skipping.")
                continue
            
            sample_counter[label] += 1
            sample_id = f"sample_{sample_counter[label]:04d}"
            
            output_dir = b_class_dir / label / sample_id
            result = process_b_class_image(img_file, yolo_model, output_dir, sample_id)
            
            if result is not None:
                result['label'] = label
                result['label_id'] = 0 if label == 'wild' else 1
                result['rgb'] = f"B_class/{label}/{sample_id}/rgb.png"
                result['mask'] = f"B_class/{label}/{sample_id}/mask.png"
                result['has_depth'] = False
                
                all_samples.append(result)
                total_processed += 1
    
    print(f"\n" + "=" * 60)
    print(f"B-class Data Processing Complete")
    print("=" * 60)
    print(f"Total processed samples: {total_processed}")
    print(f"  Wild: {sum(1 for s in all_samples if s['label'] == 'wild')}")
    print(f"  Farmed: {sum(1 for s in all_samples if s['label'] == 'farmed')}")
    
    b_class_index_file = DATASET_DIR / "B_class_index.json"
    
    b_class_index = {
        'wild': [s for s in all_samples if s['label'] == 'wild'],
        'farmed': [s for s in all_samples if s['label'] == 'farmed']
    }

    with open(b_class_index_file, 'w', encoding='utf-8') as f:
        json.dump(b_class_index, f, indent=2, ensure_ascii=False)
    
    print(f"\nB-class data index saved: {b_class_index_file}")
    print(f"Hint: Run 4_prepare_dataset.py to integrate B-class data into the final dataset.")

if __name__ == "__main__":
    main()

