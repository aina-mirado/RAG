
import redis
import json



SESSION_KEY_PREFIX = "chat_session:"
MAX_HISTORY_MESSAGES = 10

def connect_redis(): 
    try:
        pool = redis.ConnectionPool(host='localhost', port=6379, db=0, decode_responses=True)
        r = redis.Redis(connection_pool=pool)
        r.ping()

        print("Connexion Redis établie avec succès.")
    except redis.exceptions.ConnectionError as e:
        print(f"Erreur de connexion à Redis: {e}")
        print("Assurez-vous que le serveur Redis est lancé (via WSL, Docker ou service local).")
        r = None
    
    return r


r = connect_redis()

def get_session_key(session_id: str) -> str:
    return f"{SESSION_KEY_PREFIX}{session_id}"


def add_message_to_history(session_id: str, role: str, content: str):

    if r is None: return
    
    key = get_session_key(session_id)
    message = json.dumps({"role": role, "content": content})
    
    r.lpush(key, message)
    
    r.ltrim(key, 0, MAX_HISTORY_MESSAGES - 1)

def get_recent_history(session_id: str, count: int = 4):

    if r is None: return []
    
    key = get_session_key(session_id)
    raw_history = r.lrange(key, 0, count - 1)
    
    history = [json.loads(msg) for msg in raw_history]
    
    return list(reversed(history))

def get_last_q_and_a(session_id: str):
    history = get_recent_history(session_id, count=2)
    
    if len(history) < 2:
        return None
    
    previous_question = history[0]['content'] 
    previous_answer = history[1]['content']
    
    return (previous_question, previous_answer)

def clear_history(session_id: str):
    if r is None: return
    key = get_session_key(session_id)
    r.delete(key)


def run_example_redis():

    
    session_id = "test_user_session_42"
    clear_history(session_id)
    lastq = get_last_q_and_a(session_id)
    print(lastq)
    
    clear_history(session_id)
    
    add_message_to_history(session_id, "UTILISATEUR", "Ma peau me gratte...")
    add_message_to_history(session_id, "ASSISTANT", "Vous avez peut être la gale.")
    
    add_message_to_history(session_id, "UTILISATEUR", "C'est quoi la différence entre l'argent et l'or ?")
    add_message_to_history(session_id, "ASSISTANT", "L'argent est antibactérien, l'or est stable.")
    
    print("\n--- Historique Récent (4 messages) ---")
    print(get_recent_history(session_id, count=4))
    
    last_qa = get_last_q_and_a(session_id)
    print("\n--- Dernière Paire Q/R ---")
    print(f"Q: {last_qa[0]}")
    print(f"R: {last_qa[1]}")
