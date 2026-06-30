from pathlib import Path
from src.config import PATHS

def export_pipeline_to_txt():
    # 1. Retrieve target directory from configuration
    dest_dir = PATHS['llm_docs']
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Derive project base and src directories dynamically from dest_dir
    base_dir = dest_dir.parent.parent
    src_dir = base_dir / 'src'
    
    print(f"Scanning source directory: {src_dir}")
    print(f"Exporting .txt files to: {dest_dir}\n" + "-"*50)
    
    # 3. Recursively loop through all .py files inside src/ and its subfolders
    for py_path in src_dir.rglob('*.py'):
        # Skip this extraction script to avoid self-processing
        if py_path.name == 'llm.py':
            continue
            
        # Calculate relative path from src/ to preserve the folder hierarchy
        relative_path = py_path.relative_to(src_dir)
        
        # Define output path by appending .txt to the current layout (e.g., models/loss.py.txt)
        target_path = dest_dir / relative_path.with_suffix(relative_path.suffix + '.txt')
        
        # Ensure nested subdirectories exist inside docs/llm/
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Explicitly force utf-8 to safely parse comments/annotations
            content = py_path.read_text(encoding='utf-8')
            target_path.write_text(content, encoding='utf-8')
            print(f"✓ Successfully exported: src/{relative_path} -> {target_path.relative_to(base_dir)}")
        except Exception as e:
            print(f"✗ Failed to export src/{relative_path}: {e}")

if __name__ == '__main__':
    export_pipeline_to_txt()

# Run in terminal: python -m src.llm