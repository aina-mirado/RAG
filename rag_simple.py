"""
RAG SIMPLE - VERSION COURTE ET LISIBLE
Architecture: Chunking + Embedding + Retrieval + Augmentation
"""

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from data import documents, is_cache_valid
from cache_manager import CacheManager
from typing import List, Dict, Tuple


#  CONFIGURATION
# ============================================================
class Config:
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 100
    TOP_K = 2
    MIN_SIMILARITY = 0.3
    MODEL = "all-mpnet-base-v2"
    DB_PATH = "chroma_db_data"
    COLLECTION = "remedes_malgaches_fr"


#  UTILITAIRES
# ============================================================
def clean_text(text: str) -> str:
    """Nettoie le texte"""
    lines = [' '.join(line.split()) for line in text.split('\n')]
    return '\n'.join(lines).strip()


def chunk_documents(docs: List[Dict]) -> Dict:
    """Découpe les documents en chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len
    )
    
    chunked = {}
    for doc in docs:
        chunks = splitter.split_text(clean_text(doc['text']))
        chunked[doc['source']] = [
            {'text': c, 'source': doc['source']} 
            for c in chunks if c.strip()
        ]
    
    return chunked


# BASE DE DONNÉES
# ============================================================
def get_collection(reset: bool = False) -> chromadb.Collection:
    """Crée ou récupère la collection ChromaDB"""
    client = chromadb.PersistentClient(path=Config.DB_PATH)
    
    if reset:
        try:
            client.delete_collection(Config.COLLECTION)
        except:
            pass
    
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=Config.MODEL
    )
    
    return client.get_or_create_collection(
        name=Config.COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )


def store_chunks(collection: chromadb.Collection, chunked: Dict) -> int:
    """Stocke les chunks dans la BD"""
    all_ids, all_docs, all_metas = [], [], []
    
    for source, chunks in chunked.items():
        for i, chunk_data in enumerate(chunks):
            all_ids.append(f"{source}_{i}")
            all_docs.append(chunk_data['text'])
            all_metas.append({'source': source})
    
    collection.add(ids=all_ids, documents=all_docs, metadatas=all_metas)
    return len(all_ids)


#  RECHERCHE
# ============================================================
def retrieve_chunks(
    query: str,
    collection: chromadb.Collection,
    k: int = Config.TOP_K
) -> Tuple[List[str], List[str], List[float]]:
    """Récupère les chunks pertinents"""
    
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    
    if not results['documents'][0]:
        return [], [], []
    
    chunks = results['documents'][0]
    sources = [m['source'] for m in results['metadatas'][0]]
    scores = [1 - (d / 2) for d in results['distances'][0]]
    
    filtered = [
        (c, s, sc) for c, s, sc in zip(chunks, sources, scores)
        if sc >= Config.MIN_SIMILARITY
    ]
    
    if not filtered:
        return [], [], []
    
    chunks, sources, scores = zip(*filtered)
    return list(chunks), list(sources), list(scores)


#  PROMPT
# ============================================================
def build_prompt(question: str, chunks: List[str], scores: List[float]) -> str:
    """Construit le prompt RAG"""
    
    if not chunks:
        context = "Aucun contexte trouvé."
    else:
        context = "\n\n---\n\n".join(
            f"[Source {i+1} | Pertinence: {s:.0%}]\n{c}"
            for i, (c, s) in enumerate(zip(chunks, scores))
        )
    
    return f"""Vous êtes un expert en médecine malgache.

Répondez UNIQUEMENT en français et basez-vous UNIQUEMENT sur le contexte.

CONTEXTE:
{context}

QUESTION: {question}

RÉPONSE:"""


#  PIPELINE RAG
# ============================================================
class RAG:
    """Pipeline RAG complet"""
    
    def __init__(self, force_reset: bool = False):
        self.collection = get_collection(reset=force_reset)
        
        if force_reset or not is_cache_valid or self.collection.count() == 0:
            print(" Indexation des documents...")
            chunked = chunk_documents(documents)
            count = store_chunks(self.collection, chunked)
            CacheManager.update_cache("Data/pdf_documents")
            print(f"{count} chunks indexés\n")
        else:
            print(" Cache valide - Base de données réutilisée\n")
    
    def query(self, question: str) -> Dict:
        """Traite une question"""
        chunks, sources, scores = retrieve_chunks(question, self.collection)
        prompt = build_prompt(question, chunks, scores)
        
        return {
            'prompt': prompt,
            'chunks': chunks,
            'sources': sources,
            'scores': scores,
            'count': len(chunks)
        }
    
    def stats(self) -> Dict:
        """Statistiques"""
        return {
            'chunks': self.collection.count(),
            'model': Config.MODEL
        }


#  MAIN - TESTS
# ============================================================
if __name__ == "__main__":
    print("\nRAG - TEST\n")
    
    # Initialiser
    rag = RAG(force_reset=False)
    
    # Stats
    stats = rag.stats()
    print(f" Stats: {stats['chunks']} chunks indexés\n")
    
    # Test requête
    question = "quelle est le médicament pour le grippe?"
    result = rag.query(question)
    
    print(f" Question: {question}")
    print(f" Résultats: {result['count']}")
    print(f"  Sources: {', '.join(set(result['sources']))}")
    print(f" Scores: {[f'{s:.0%}' for s in result['scores']]}\n")
    print(f" Prompt généré:\n{result['prompt']}\n")