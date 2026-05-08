import os

from pathlib import Path
from typing import List, Dict
import pdfplumber
from cache_manager import CacheManager


# CONFIG
PDF_FOLDER = "Data/pdf_documents"
MIN_TEXT_LENGTH = 50


# EXTRACTION
def clean_text(text: str) -> str:
    """Nettoie et normalise le texte"""
    if not text:
        return ""
    
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        line = ' '.join(line.split())
        
        if line and not (line.isdigit() and len(line) <= 3):
            lines.append(line)
    
    result = '\n'.join(lines)
    while '\n\n\n' in result:
        result = result.replace('\n\n\n', '\n\n')
    
    return result.strip()


def extract_from_pdf(pdf_path: str) -> tuple:
    """Extrait texte et métadonnées d'un PDF"""
    try:
        text_parts = []
        page_count = 0
        table_count = 0
        
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(clean_text(text))
                
                tables = page.extract_tables()
                if tables:
                    table_count += len(tables)
                    for table in tables:
                        rows = [' | '.join(str(c) if c else '' for c in row) for row in table]
                        text_parts.append('TABLE:\n' + '\n'.join(rows))
        
        full_text = '\n\n'.join(text_parts)
        metadata = {
            'pages': page_count,
            'tables': table_count,
            'chars': len(full_text)
        }
        
        return full_text, metadata
    
    except Exception as e:
        raise Exception(f"Erreur extraction {pdf_path}: {e}")


def load_documents() -> List[Dict]:
    """Charge tous les PDFs du dossier"""
    folder = Path(PDF_FOLDER)
    
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        return []
    
    pdf_files = list(folder.glob("*.pdf"))
    
    if not pdf_files:
        return []
    
    documents = []
    
    for pdf_path in sorted(pdf_files):
        try:
            text, metadata = extract_from_pdf(str(pdf_path))
            
            if len(text) >= MIN_TEXT_LENGTH:
                documents.append({
                    'source': pdf_path.name,
                    'text': text,
                    'metadata': metadata
                })
        
        except Exception as e:
            print(f"Erreur {pdf_path.name}: {e}")
    
    return documents


# EXPORT
documents = load_documents()
is_cache_valid = CacheManager.is_cache_valid(PDF_FOLDER)