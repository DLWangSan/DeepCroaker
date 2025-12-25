"""
Check dataset integrity: verify all required files exist and output statistics
"""
import json
from pathlib import Path
from collections import defaultdict

CONFIG_FILE = Path(__file__).parent / "config.json"

def load_config():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"配置文件不存在: {CONFIG_FILE}")
    
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

config = load_config()
DATASET_DIR = Path(config['paths']['dataset_dir'])

def check_sample_files(sample_record, dataset_dir):
    """Check if required files for a sample exist"""
    missing_files = []
    
    rgb_path = dataset_dir / sample_record['rgb']
    if not rgb_path.exists():
        missing_files.append(sample_record['rgb'])
    
    if 'mask' in sample_record:
        mask_path = dataset_dir / sample_record['mask']
        if not mask_path.exists():
            missing_files.append(sample_record['mask'])
    
    if sample_record.get('has_depth', False):
        if 'depth' in sample_record:
            depth_path = dataset_dir / sample_record['depth']
            if not depth_path.exists():
                missing_files.append(sample_record['depth'])
    
    is_valid = len(missing_files) == 0
    return is_valid, missing_files

def check_dataset(dataset_dir):
    """Check dataset integrity"""
    dataset_dir = Path(dataset_dir)
    
    print("=" * 60)
    print("Dataset Integrity Check")
    print("=" * 60)
    print(f"Dataset directory: {dataset_dir}")
    
    index_file = dataset_dir / "dataset_index.json"
    if not index_file.exists():
        print(f"\nError: dataset index file not found: {index_file}")
        print("   Please run: python 4_prepare_dataset.py")
        return
    
    print(f"Index file exists: {index_file}")
    
    with open(index_file, 'r', encoding='utf-8') as f:
        dataset_index = json.load(f)
    stats = {
        'A_class': {
            'train': {'wild': {'total': 0, 'valid': 0, 'invalid': 0}, 
                     'farmed': {'total': 0, 'valid': 0, 'invalid': 0}},
            'val': {'wild': {'total': 0, 'valid': 0, 'invalid': 0}, 
                   'farmed': {'total': 0, 'valid': 0, 'invalid': 0}}
        },
        'B_class': {
            'train': {'wild': {'total': 0, 'valid': 0, 'invalid': 0}, 
                     'farmed': {'total': 0, 'valid': 0, 'invalid': 0}},
            'val': {'wild': {'total': 0, 'valid': 0, 'invalid': 0}, 
                   'farmed': {'total': 0, 'valid': 0, 'invalid': 0}}
        }
    }
    
    invalid_samples = []
    
    print("\nChecking sample files...")
    for data_class in ['A_class', 'B_class']:
        for split in ['train', 'val']:
            for label in ['wild', 'farmed']:
                samples = dataset_index[data_class][split][label]
                
                for sample in samples:
                    stats[data_class][split][label]['total'] += 1
                    
                    is_valid, missing_files = check_sample_files(sample, dataset_dir)
                    
                    if is_valid:
                        stats[data_class][split][label]['valid'] += 1
                    else:
                        stats[data_class][split][label]['invalid'] += 1
                        invalid_samples.append({
                            'sample_id': sample.get('sample_id', 'unknown'),
                            'class': data_class,
                            'split': split,
                            'label': label,
                            'missing_files': missing_files
                        })
    
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    print("\n[A-class data (structured light data, with depth)]")
    a_train_wild = stats['A_class']['train']['wild']
    a_train_farmed = stats['A_class']['train']['farmed']
    a_val_wild = stats['A_class']['val']['wild']
    a_val_farmed = stats['A_class']['val']['farmed']
    
    print("\n  Train set:")
    print(f"    Wild: {a_train_wild['valid']}/{a_train_wild['total']} valid")
    if a_train_wild['invalid'] > 0:
        print(f"          {a_train_wild['invalid']} samples missing files")
    print(f"    Farmed: {a_train_farmed['valid']}/{a_train_farmed['total']} valid")
    if a_train_farmed['invalid'] > 0:
        print(f"          {a_train_farmed['invalid']} samples missing files")
    print(f"    Subtotal: {a_train_wild['valid'] + a_train_farmed['valid']}/{a_train_wild['total'] + a_train_farmed['total']} valid")
    
    print("\n  Validation set:")
    print(f"    Wild: {a_val_wild['valid']}/{a_val_wild['total']} valid")
    if a_val_wild['invalid'] > 0:
        print(f"          {a_val_wild['invalid']} samples missing files")
    print(f"    Farmed: {a_val_farmed['valid']}/{a_val_farmed['total']} valid")
    if a_val_farmed['invalid'] > 0:
        print(f"          {a_val_farmed['invalid']} samples missing files")
    print(f"    Subtotal: {a_val_wild['valid'] + a_val_farmed['valid']}/{a_val_wild['total'] + a_val_farmed['total']} valid")
    
    a_total = (a_train_wild['total'] + a_train_farmed['total'] + 
               a_val_wild['total'] + a_val_farmed['total'])
    a_valid = (a_train_wild['valid'] + a_train_farmed['valid'] + 
               a_val_wild['valid'] + a_val_farmed['valid'])
    print(f"\n  A-class Total: {a_valid}/{a_total} valid")
    
    print("\n[B-class data (action camera data, no depth)]")
    b_train_wild = stats['B_class']['train']['wild']
    b_train_farmed = stats['B_class']['train']['farmed']
    b_val_wild = stats['B_class']['val']['wild']
    b_val_farmed = stats['B_class']['val']['farmed']
    
    print("\n  Train set:")
    print(f"    Wild: {b_train_wild['valid']}/{b_train_wild['total']} valid")
    if b_train_wild['invalid'] > 0:
        print(f"          {b_train_wild['invalid']} samples missing files")
    print(f"    Farmed: {b_train_farmed['valid']}/{b_train_farmed['total']} valid")
    if b_train_farmed['invalid'] > 0:
        print(f"          {b_train_farmed['invalid']} samples missing files")
    print(f"    Subtotal: {b_train_wild['valid'] + b_train_farmed['valid']}/{b_train_wild['total'] + b_train_farmed['total']} valid")
    
    print("\n  Validation set:")
    print(f"    Wild: {b_val_wild['valid']}/{b_val_wild['total']} valid")
    if b_val_wild['invalid'] > 0:
        print(f"          {b_val_wild['invalid']} samples missing files")
    print(f"    Farmed: {b_val_farmed['valid']}/{b_val_farmed['total']} valid")
    if b_val_farmed['invalid'] > 0:
        print(f"          {b_val_farmed['invalid']} samples missing files")
    print(f"    Subtotal: {b_val_wild['valid'] + b_val_farmed['valid']}/{b_val_wild['total'] + b_val_farmed['total']} valid")
    
    b_total = (b_train_wild['total'] + b_train_farmed['total'] + 
               b_val_wild['total'] + b_val_farmed['total'])
    b_valid = (b_train_wild['valid'] + b_train_farmed['valid'] + 
               b_val_wild['valid'] + b_val_farmed['valid'])
    print(f"\n  B-class Total: {b_valid}/{b_total} valid")
    
    print("\n" + "=" * 60)
    print("Overall Statistics")
    print("=" * 60)
    total_samples = a_total + b_total
    total_valid = a_valid + b_valid
    total_invalid = total_samples - total_valid
    
    print(f"Total samples: {total_samples}")
    print(f"Valid samples: {total_valid} ({total_valid/total_samples*100:.1f}%)")
    print(f"Invalid samples: {total_invalid} ({total_invalid/total_samples*100:.1f}%)")
    
    train_total = (a_train_wild['total'] + a_train_farmed['total'] + 
                   b_train_wild['total'] + b_train_farmed['total'])
    train_valid = (a_train_wild['valid'] + a_train_farmed['valid'] + 
                   b_train_wild['valid'] + b_train_farmed['valid'])
    val_total = (a_val_wild['total'] + a_val_farmed['total'] + 
                 b_val_wild['total'] + b_val_farmed['total'])
    val_valid = (a_val_wild['valid'] + a_val_farmed['valid'] + 
                 b_val_wild['valid'] + b_val_farmed['valid'])
    
    print(f"\nTrain set: {train_valid}/{train_total} valid ({train_valid/train_total*100:.1f}%)")
    print(f"Validation set: {val_valid}/{val_total} valid ({val_valid/val_total*100:.1f}%)")
    
    if invalid_samples:
        print("\n" + "=" * 60)
        print("Invalid Sample Details (first 20)")
        print("=" * 60)
        for i, sample in enumerate(invalid_samples[:20], 1):
            print(f"\n{i}. {sample['sample_id']} ({sample['class']}, {sample['split']}, {sample['label']})")
            print(f"   Missing files:")
            for missing_file in sample['missing_files']:
                print(f"     - {missing_file}")
        
        if len(invalid_samples) > 20:
            print(f"\n... {len(invalid_samples) - 20} more invalid samples")
    
    print("\n" + "=" * 60)
    if total_invalid == 0:
        print("Dataset check completed: all sample files are intact!")
    else:
        print(f"Dataset check completed: found {total_invalid} invalid samples")
        print("   Please check the missing files above, or re-run data preprocessing scripts")
    print("=" * 60)

def main():
    try:
        check_dataset(DATASET_DIR)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

