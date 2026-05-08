import rag

def list_chunks_simple(collection_name="remedes_malgaches_fr", CHROMA_PATH="chroma_db_data"):
    """Affichage simple des chunks"""
    try:
        client = rag.init_chromadb_client(CHROMA_PATH)
        collection = client.get_collection(collection_name)
        all_data = collection.get()
        
        print(f"\n✅ {len(all_data['documents'])} chunks trouvés:\n")
        
        for i, (chunk_id, text) in enumerate(zip(all_data['ids'], all_data['documents']), 1):
            print(f"[{i}] {chunk_id}: {text}...\n")
    
    except Exception as e:
        print(f"❌ Erreur: {e}")

def count_chunks(collection_name="remedes_malgaches_fr", CHROMA_PATH="chroma_db_data"):
    """Compte le nombre total de chunks"""
    try:
        client = rag.init_chromadb_client(CHROMA_PATH)
        collection = client.get_collection(collection_name)
        count = collection.count()
        print(f"\n📊 Nombre total de chunks: {count}\n")
        return count
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return 0

