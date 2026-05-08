import json
import hashlib
from pathlib import Path
from typing import Dict, Optional


class CacheManager:
    CACHE_FILE = ".cache_metadata.json"
    
    @staticmethod
    def get_file_hash(filepath: str) -> str:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    @staticmethod
    def get_folder_hash(folder: str) -> str:
        folder_path = Path(folder)
        hashes = []
        
        for pdf_file in sorted(folder_path.glob("*.pdf")):
            hashes.append(CacheManager.get_file_hash(str(pdf_file)))
        
        combined = ''.join(hashes)
        return hashlib.md5(combined.encode()).hexdigest()
    
    @staticmethod
    def load_cache() -> Optional[Dict]:
        if Path(CacheManager.CACHE_FILE).exists():
            with open(CacheManager.CACHE_FILE, 'r') as f:
                return json.load(f)
        return None
    
    @staticmethod
    def save_cache(data: Dict) -> None:
        with open(CacheManager.CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def is_cache_valid(pdf_folder: str) -> bool:
        cache = CacheManager.load_cache()
        if not cache:
            return False
        
        current_hash = CacheManager.get_folder_hash(pdf_folder)
        return cache.get('folder_hash') == current_hash
    
    @staticmethod
    def update_cache(pdf_folder: str) -> None:
        cache = {
            'folder_hash': CacheManager.get_folder_hash(pdf_folder),
            'status': 'indexed'
        }
        CacheManager.save_cache(cache)
    
    @staticmethod
    def invalidate_cache() -> None:
        if Path(CacheManager.CACHE_FILE).exists():
            Path(CacheManager.CACHE_FILE).unlink()