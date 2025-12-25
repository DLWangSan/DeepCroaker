"""
Run all preprocessing steps in batch
"""
import sys
from pathlib import Path

def run_step(script_name, description):
    """Run a single step"""
    print("\n" + "=" * 60)
    print(description)
    print("=" * 60)
    
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"Error: script not found: {script_path}")
        return False
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("script", script_path)
    module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(module)
        if hasattr(module, 'main'):
            module.main()
        return True
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Data Preprocessing - Batch Run")
    print("=" * 60)
    
    steps = [
        ("3_generate_depth_maps_with_yolo.py", "Step 1: Generate depth maps and masks (using YOLO)"),
        ("4_prepare_dataset.py", "Step 2: Prepare final dataset"),
    ]
    
    print("Note: Assumes data has been processed through step 0 (scan and organize)")
    print("      If re-scanning is needed, manually run: 1_scan_and_organize_data.py")
    
    if len(sys.argv) > 1 and sys.argv[1] == '--skip-questions':
        skip_questions = True
    else:
        skip_questions = False
        print("\nWill run the following steps in order:")
        for i, (script, desc) in enumerate(steps, 1):
            print(f"  {i}. {desc}")
        print("\nTip: Use --skip-questions to skip confirmation")
        
        response = input("\nContinue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled")
            return
    
    for script, description in steps:
        success = run_step(script, description)
        if not success:
            print(f"\nStep failed: {description}")
            response = input("Continue with next steps? (y/n): ")
            if response.lower() != 'y':
                break
    
    print("\n" + "=" * 60)
    print("Preprocessing completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Check G:\\dhy_data\\ directory")
    print("  2. View G:\\dhy_data\\dataset_index.json")
    print("  3. Start model training")

if __name__ == "__main__":
    main()

