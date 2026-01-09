import hashlib
import json
import logging
import os
import tempfile
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Constants
CHUNK_SIZE = 8192  # 8KB chunks for reading files
DEFAULT_FILE_EXTENSION = '.pdf'

class HashProcessor:

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if not verbose:
            self.logger.setLevel(logging.WARNING)
    
    def get_file_hash(self, path: str) -> Optional[str]:
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while chunk := f.read(CHUNK_SIZE):
                    h.update(chunk)
            return h.hexdigest()
        except (IOError, OSError) as e:
            self.logger.error(f"Error reading file {path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unknown error processing file {path}: {e}")
            return None
    
    def scan_files_for_hash(
        self, 
        source_dir: str, 
        file_extension: str = DEFAULT_FILE_EXTENSION,
        recursive: bool = False
    ) -> Dict[str, List[Dict[str, str]]]:
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Directory not found: {source_dir}")
            
        hash_to_files = defaultdict(list)
        self.logger.info(f"Scanning files in: {source_dir}")
        
        pattern = f"**/*{file_extension}" if recursive else f"*{file_extension}"
        
        try:
            files = list(source_path.glob(pattern))
            
            for file_path in files:
                if not file_path.is_file():
                    continue
                    
                self.logger.info(f"Calculating hash for: {file_path.name}")
                
                file_hash = self.get_file_hash(str(file_path))
                if file_hash:
                    hash_to_files[file_hash].append({
                        'filename': file_path.name,
                        'path': str(file_path),
                        'size': file_path.stat().st_size
                    })
        except PermissionError as e:
            self.logger.error(f"Permission error: {e}")
            raise
        
        return hash_to_files
    
    def load_processed_index(self, index_file: str) -> Dict:
        if os.path.exists(index_file):
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error reading index file {index_file}: {e}")
                return {}
            except Exception as e:
                self.logger.error(f"Unknown error reading index: {e}")
                return {}
        return {}
    
    def save_processed_index(self, index_file: str, processed_hashes: Dict) -> None:
        temp_name = None
        try:
            os.makedirs(os.path.dirname(index_file), exist_ok=True)
            
            # Write to temp file first
            dir_name = os.path.dirname(index_file)
            with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, encoding='utf-8') as tmp_file:
                json.dump(processed_hashes, tmp_file, indent=2, ensure_ascii=False)
                temp_name = tmp_file.name
            
            # Rename temp file to main file (atomic operation on POSIX)
            shutil.move(temp_name, index_file)
            self.logger.info(f"Index file saved safely: {index_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving index file {index_file}: {e}")
            if temp_name and os.path.exists(temp_name):
                os.remove(temp_name)
    
    def get_current_timestamp(self) -> str:
        return datetime.now().isoformat()
    
    def get_string_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

