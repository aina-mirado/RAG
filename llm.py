from transformers import pipeline
from rag import retrieve_chunks, get_rag  #  Ajouter get_rag


def transform_question(last_question, previous_question, previous_answer):
          
    generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    device=0,
    do_sample=False
    )

    prompt = f"""
Sign in
Use a different account


    given a chat history and user question which might reference context in the chat history,
    formulate a standalone question which can be understood without the chat history.
    Do NOT answer the question , just reformulate it if needed and otheriwise return it as is

    History :
    - Q: {previous_question}
    - R: {previous_answer}
    
    new question : {last_question}
    
    [rewritten question]:
    """

    
    try:
        result = generator(
            prompt,
            max_new_tokens=64
        )

        
        generated_text = result[0]['generated_text'].strip()

        return generated_text

    except Exception as e:
        return f"ERREUR PIPELINE : {e}"
    


def generate_response_rag_pipeline(user_query_fr: str):
    relevant_chunks, sources = retrieve_chunks(user_query_fr, k=2)  # k=2 au lieu de k=3
    context_str = "\n".join(relevant_chunks)

    prompt = f"""
    **[Instruction Système]**
    Vous êtes un assistant expert en synthèse. Répondez à la question de l'utilisateur **UNIQUEMENT** en vous basant sur le CONTEXTE fourni ci-dessous.

    **Règles de Réponse :**
    1.  **Synthèse :** Reformulez et synthétisez les informations de manière concise et précise.
    2.  **Honnêteté :** Si l'information complète pour répondre à la question est absente du CONTEXTE, répondez simplement : "**L'information n'est pas disponible dans le contexte fourni.**"
    3.  **Format :** Ne faites aucune introduction ni conclusion. Allez directement à la réponse.
    
    **[CONTEXTE]**
    {context_str}
    
    **[QUESTION]**
    {user_query_fr}
    """

    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        device=0
    )
    
    try:
        result = generator(
            prompt,
            max_new_tokens=64
        )

        generated_text = result[0]['generated_text'].strip()

        return generated_text

    except Exception as e:
        return f"ERREUR PIPELINE : {e}"