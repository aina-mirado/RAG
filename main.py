# main.py
from fastapi import FastAPI
import datetime
import os
import json
import llm
import redis_manager
from rag import get_rag  # CHANGEMENT: Ajouter cet import


app = FastAPI()


@app.get("/chat-meditrad/")
def respond_user(user_id: str, msg: str):
    history = redis_manager.get_last_q_and_a(user_id)
    redis_manager.clear_history(user_id)

    if history:
        prev_q , prev_a = history
        transformed_question = llm.transform_question(msg, prev_q, prev_a)
    else:
        transformed_question = msg
    
    print("l'historique est: " , history)
    print("\n   ")
    print("La question à chercher la reponse est: ", transformed_question)
    response = llm.generate_response_rag_pipeline(transformed_question)

    redis_manager.add_message_to_history(user_id, "user", msg)
    redis_manager.add_message_to_history(user_id, "chatRAG", response)

    return {"response" : response}


@app.get("/")
def home():
    return {"message": "API est en marche! Allez à /docs pour voir les routes disponibles."}


@app.get("/stats/")  # OPTIONNEL: Endpoint de stats
def stats():
    rag = get_rag()
    return {"chunks": rag.stats()['chunks']}