"""
Scan and organize A-class data (structured light data)
"""
import os
import shutil
from pathlib import Path
from collections import defaultdict
import json

CONFIG_FILE = Path(__file__).parent / "config.json"

def load_config():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"配置文件不存在: {CONFIG_FILE}")
    
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

config = load_config()
SOURCE_DIR = config['paths']['source_dir']
ORGANIZED_DIR = config['paths']['organized_dir']
EXCLUDE_FOLDERS = config['processing']['exclude_folders']

def scan_data_structure(source_dir):
    """Scan data directory structure"""
    data_files = defaultdict(lambda: defaultdict(list))
    
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Error: source directory not found: {source_dir}")
        return data_files
    
    for folder in source_path.iterdir():
        if not folder.is_dir():
            continue
            
        if any(exclude in folder.name for exclude in EXCLUDE_FOLDERS):
            print(f"Skipping excluded folder: {folder.name}")
            continue
        
        print(f"Processing folder: {folder.name}")
        
        label = None
        if "野生" in folder.name:
            label = "wild"
        elif "养殖" in folder.name:
            label = "farmed"
        else:
            print(f"  Warning: cannot determine label, skipping: {folder.name}")
            continue
        
        for file_path in folder.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in ['.ply', '.png']:
                    rel_path = file_path.relative_to(folder)
                    base_name = rel_path.stem
                    key = f"{label}_{folder.name}_{base_name}"
                    
                    data_files[key][ext[1:]].append({
                        'path': str(file_path),
                        'type': ext[1:],
                        'base_name': base_name,
                        'label': label,
                        'folder': folder.name
                    })
    
    return data_files

def organize_data(data_files, output_dir):
    """Organize data to unified directory structure"""
    organized = {
        'wild': [],
        'farmed': []
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for label in ['wild', 'farmed']:
        (output_path / label).mkdir(exist_ok=True)
    
    for key, file_dict in data_files.items():
        has_rgb = 'png' in file_dict and len(file_dict['png']) > 0
        has_pointcloud = 'ply' in file_dict and len(file_dict['ply']) > 0
        
        if not (has_rgb and has_pointcloud):
            parts = key.split('_', 2)
            if len(parts) >= 2:
                base_name = parts[-1] if len(parts) > 2 else 'unknown'
                print(f"  Warning: {base_name} missing RGB or point cloud, skipping")
            continue
        
        first_file = None
        for file_list in file_dict.values():
            if file_list:
                first_file = file_list[0]
                break
        
        if first_file is None:
            continue
        
        label = first_file['label']
        sample_id = len(organized[label]) + 1
        sample_dir = output_path / label / f"sample_{sample_id:04d}"
        sample_dir.mkdir(exist_ok=True)
        
        sample_info = {
            'sample_id': f"sample_{sample_id:04d}",
            'label': label,
            'source_key': key,
            'base_name': first_file['base_name'],
            'files': {}
        }
        
        if 'png' in file_dict and file_dict['png']:
            rgb_file = file_dict['png'][0]
            dest_rgb = sample_dir / "rgb.png"
            shutil.copy2(rgb_file['path'], dest_rgb)
            sample_info['files']['rgb'] = str(dest_rgb.relative_to(output_path))
        
        if 'ply' in file_dict and file_dict['ply']:
            ply_file = file_dict['ply'][0]
            dest_ply = sample_dir / "pointcloud.ply"
            shutil.copy2(ply_file['path'], dest_ply)
            sample_info['files']['pointcloud'] = str(dest_ply.relative_to(output_path))
            sample_info['files']['format'] = 'ply'
        
        organized[label].append(sample_info)
    
    return organized

def main():
    print("=" * 60)
    print("Step 1: Scan data directory structure")
    print("=" * 60)
    
    data_files = scan_data_structure(SOURCE_DIR)
    
    print(f"\nFound {len(data_files)} potential samples")
    print(f"  Wild: {sum(1 for k in data_files.keys() if k.startswith('wild_'))}")
    print(f"  Farmed: {sum(1 for k in data_files.keys() if k.startswith('farmed_'))}")
    
    print("\n" + "=" * 60)
    print("Step 2: Organize data to unified directory")
    print("=" * 60)
    
    organized = organize_data(data_files, ORGANIZED_DIR)
    
    index_file = Path(ORGANIZED_DIR) / "data_index.json"
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(organized, f, indent=2, ensure_ascii=False)
    
    print(f"\nDone!")
    print(f"  Wild samples: {len(organized['wild'])}")
    print(f"  Farmed samples: {len(organized['farmed'])}")
    print(f"  Total: {len(organized['wild']) + len(organized['farmed'])}")
    print(f"\nIndex file saved: {index_file}")

if __name__ == "__main__":
    main()

