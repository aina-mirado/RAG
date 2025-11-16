from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

corpus_francais = """
Le Ravintsara (Cinnamomum camphora) est une plante endémique de Madagascar très prisée en médecine traditionnelle pour ses vertus antivirales et immunostimulantes.
Ses feuilles sont principalement utilisées pour traiter les affections respiratoires courantes telles que le rhume, la grippe, la bronchite et la toux.
Il est essentiel de le distinguer du Camphrier (Cinnamomum camphora) dont l'huile essentielle est très différente.
Préparation du Ravintsara : Pour une infusion, faites bouillir 3 à 5 feuilles fraîches ou séchées dans 250 ml d'eau pendant 10 minutes.
Cette boisson est traditionnellement consommée trois fois par jour.
Contre-indications : L'usage du Ravintsara, en particulier de l'huile essentielle, doit être évité chez la femme enceinte et le nourrisson de moins de trois ans.
Il ne remplace en aucun cas un avis médical.

Le Kininina (Cinchona officinalis) est célèbre pour son écorce contenant de la quinine, un puissant antipaludéen historiquement utilisé contre la fièvre et le paludisme.
C'est un remède incontournable en cas de fièvre élevée ou de fortes douleurs articulaires.
Préparation du Kininina : L'écorce est souvent utilisée en décoction. Faites bouillir 10 grammes d'écorce séchée par litre d'eau pendant 20 minutes, puis filtrez.
Le liquide est amer et doit être consommé avec prudence.
Avertissement : La surconsommation de Kininina peut entraîner des troubles auditifs. Le dosage doit être strict, et l'auto-médication prolongée est déconseillée.
Il est strictement interdit aux personnes ayant des problèmes cardiaques.
"""

test_text = """Doura était l’un des étudiants les plus populaires de l’université. Grand, 
charismatique et toujours souriant, il attirait les regards où qu’il aille. Dès sa première année, 
il avait acquis une réputation : il était le séducteur incontesté du campus. 
Peu importait la faculté ou le club auquel il participait, 
les filles semblaient toujours s’intéresser à lui, et Doura, 
amusé par cette attention constante, profitait de chaque moment. 
Il enchaînait les rendez-vous, les sorties et les messages charmeurs, convaincu que personne ne pourrait jamais l’attacher.
Cependant, derrière ce sourire confiant se cachait une certaine solitude. 
Doura aimait la compagnie, mais il ne connaissait pas encore la profondeur d’une véritable connexion 
émotionnelle. Ses relations étaient souvent éphémères, basées sur le charme et l’attraction, 
mais jamais sur la sincérité ou l’engagement. Il se plaisait à croire que l’amour véritable n’était 
qu’un mythe, ou du moins, quelque chose qu’il n’avait pas encore rencontré.
Tout changea un matin d’automne. Alors qu’il se rendait à la bibliothèque pour réviser un examen 
de philosophie, Doura aperçut une nouvelle étudiante, assise dans un coin tranquille, 
plongée dans un livre de poésie. Elle avait l’air absorbée par ses lectures et indifférente
au tumulte autour d’elle. Intrigué, Doura s’approcha pour lui adresser un simple « bonjour ».
À sa grande surprise, elle ne réagit pas de la manière attendue. Plutôt que d’être séduite
ou intimidée, elle le regarda calmement et lui sourit avec une chaleur sincère. Ce sourire,
si authentique, fit vaciller quelque chose en lui.
Son nom était Élise. Contrairement aux autres étudiantes, elle n’était pas impressionnée par 
son statut de populaire. Elle était intelligente, drôle et passionnée par la littérature et
les arts, et surtout, elle semblait voir au-delà du charme de Doura. Chaque rencontre avec 
elle devenait une découverte : ils parlaient pendant des heures, partageant leurs idées,
leurs rêves et parfois leurs peurs. Doura se rendit compte qu’il avait enfin trouvé quelqu’un qui 
le comprenait vraiment.
Petit à petit, il commença à changer. Ses anciennes habitudes de séducteur lui semblaient
désormais vides et superficielles. Il n’avait plus envie de séduire pour le plaisir ;
son cœur était désormais captif de quelque chose de plus profond et de plus vrai.
Pour la première fois, Doura comprit ce que signifiait aimer quelqu’un sincèrement 
et être aimé en retour, sans artifices ni jeux.
Leurs amis remarquèrent le changement. Le garçon qui autrefois enchaînait les conquêtes était
devenu attentionné, patient et doux. Il avait trouvé dans Élise un équilibre qu’il n’avait jamais 
connu auparavant. Même les moments les plus simples — un café partagé à la cafétéria, une balade sur le campus au coucher du soleil — devenaient pour lui précieux et significatifs.
Doura avait enfin compris que la popularité et les conquêtes étaient éphémères, mais que 
le véritable amour, celui qui touche l’âme et transforme le cœur, était rare et inestimable.
Et c’est ainsi qu’un étudiant autrefois connu pour son charme superficiel devint un jeune homme
capable d’aimer profondément, avec la certitude qu’Élise était bien plus qu’une simple rencontre :
elle était son vrai amour."""


documents = [
    {"source": "Ravintsara (C. camphora)", "text": corpus_francais.split("Le Kininina")[0].strip()},
    {"source": "Kininina (C. officinalis)", "text": "Le Kininina" + corpus_francais.split("Le Kininina")[1].strip()},
    {"source": "Doura", "text": test_text}
]

def load_collection(collection_name = "remedes_malgaches_fr", CHROMA_PATH = "chroma_db_data", model_name = 'all-MiniLM-L6-v2', create = False):

    chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    if create:
        try:
            chroma_client.delete_collection(name=collection_name)
            print(f"Collection '{collection_name}' précédente supprimée.")
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=chroma_ef
    )
    
    return collection

def add_collection(content, collection_name = "remedes_malgaches_fr", CHROMA_PATH = "chroma_db_data", model_name = 'all-MiniLM-L6-v2', create = False):
    
    title = content['source']
    text = content['text']

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100,
    length_function=len
    )
     
    all_chunks = text_splitter.split_text(text.strip())
    all_metadatas =  [{"source": title}] * len(all_chunks)
    
    print(f"Le corpus a été découpé en {len(all_chunks)} morceaux (chunks).")

    collection = load_collection(create=create)
            
    ids = [f"{title}_{i}" for i in range(len(all_chunks))]
    
    collection.add(
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=ids
    )


def retrieve_chunks(query: str, k: int = 3):
    collection = load_collection()
    
    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    relevant_chunks = results['documents'][0]
    relevant_sources = [meta['source'] for meta in results['metadatas'][0]]

    return relevant_chunks, relevant_sources