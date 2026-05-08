"""
MAIN_TERMINAL.PY - CHATBOT INTERACTIF EN TERMINAL
Interaction directe sans API FastAPI
"""

import time
import llm
import redis_manager


def main():
    user_id = input("👤 User ID (défaut 'user_1'): ").strip() or "user_1"
    
    print("\n" + "="*60)
    print("🤖 CHATBOT MÉDECINE MALGACHE".center(60))
    print("="*60 + "\n")
    
    rag = llm.get_rag()  # ✅ Fonctionne maintenant
    print(f"✅ RAG prêt ({rag.stats()['chunks']} chunks)\n")
    
    while True:
        try:
            question = input("❓ Question (exit pour quitter): ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        
        if not question:
            continue
        
        if question.lower() in ("exit", "quit"):
            break
        
        history = redis_manager.get_last_q_and_a(user_id)
        
        if history:
            prev_q, prev_a = history
            transformed = llm.transform_question(question, prev_q, prev_a)
        else:
            transformed = question
        
        print("\n⏳ Génération...\n")
        start = time.time()
        response = llm.generate_response_rag_pipeline(transformed)
        duration = time.time() - start
        
        print(f"✅ ({duration:.1f}s):\n{response}\n")
        
        redis_manager.add_message_to_history(user_id, "user", question)
        redis_manager.add_message_to_history(user_id, "chatRAG", response)
    
    print("\n👋 Bye!\n")


if __name__ == "__main__":
    main()