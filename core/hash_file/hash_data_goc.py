import sys
import os
import json
import shutil
from pathlib import Path

# Setup path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.hash_file.hash_file import HashProcessor

def main():
    # Initialize
    data_dir = project_root / "data"
    files_dir = data_dir / "files" 
    files_dir.mkdir(parents=True, exist_ok=True)
    
    # Local data source path
    source_root = Path("/home/bahung/Do_An_Dataset/data_rag/")
    if not source_root.exists():
        print(f"Source directory not found: {source_root}")
        return

    hash_processor = HashProcessor(verbose=False)
    hash_file_path = data_dir / "hash_data_goc_index.json"
    
    # Load existing hash index
    existing_hashes = {}
    if hash_file_path.exists():
        with open(hash_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            existing_hashes = {item['filename']: item['hash'] for item in data.get('train', [])}
        print(f"Loaded {len(existing_hashes)} hashes from old index")
    
    print(f"Scanning files from: {source_root}")
    
    # Find all PDF files in source directory (recursive)
    pdf_files = list(source_root.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files\n")
    
    # Process each file
    hash_results = []
    skipped = 0
    processed = 0
    
    for idx, source_path in enumerate(pdf_files):
        # Calculate relative path to preserve directory structure
        relative_path = source_path.relative_to(source_root)
        filename = str(relative_path)
        
        # Destination path
        dest_path = files_dir / relative_path
        
        # Create parent directory at destination if not exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists and hash matches
        if dest_path.exists() and filename in existing_hashes:
            current_hash = hash_processor.get_file_hash(str(dest_path))
            if current_hash == existing_hashes[filename]:
                hash_results.append({
                    'filename': filename,
                    'hash': current_hash,
                    'index': idx
                })
                skipped += 1
                continue
        
        try:
            # Copy file from source to destination
            shutil.copy2(source_path, dest_path)
            
            # Calculate hash
            file_hash = hash_processor.get_file_hash(str(dest_path))
            if file_hash is None:
                print(f"Error calculating hash for file {filename}")
                continue
            
            hash_results.append({
                'filename': filename,
                'hash': file_hash,
                'index': idx
            })
            processed += 1
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(pdf_files)} files")
                
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue

    # Save results
    output_data = {
        'train': hash_results,
        'total_files': len(hash_results)
    }
    
    with open(hash_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    print(f"Done!")
    print(f"   - Total files: {len(hash_results)}")
    print(f"   - Newly processed: {processed}")
    print(f"   - Skipped (hash match): {skipped}")
    print(f"   - Index file: {hash_file_path}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
