"""
RAG.PY - WRAPPER POUR COMPATIBILITÉ AVEC LLM.PY
Permet d'utiliser rag_simple sans changer llm.py
"""

from rag_simple import RAG, Config, retrieve_chunks as _retrieve_chunks
from typing import Tuple, List, Optional


_rag_instance: Optional[RAG] = None


def get_rag() -> RAG:
    """Récupère l'instance RAG globale (singleton)"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAG(force_reset=False)
    return _rag_instance


def retrieve_chunks(query: str, k: int = 3) -> Tuple[List[str], List[str]]:
    """
    API compatible avec ancien code
    Retourne: (chunks, sources)
    """
    rag = get_rag()
    result = rag.query(query)
    
    return result['chunks'], result['sources']