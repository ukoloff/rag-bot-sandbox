from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction, DefaultEmbeddingFunction
from langchain_gigachat import GigaChatEmbeddings
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
import numpy as np
import chromadb
from os.path import join, dirname, normpath

class GigaChatEmb(EmbeddingFunction):

    def __init__(self, model_name='Embeddings'):
        super().__init__()
        self.giga = GigaChatEmbeddings(model=model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return [np.array(vec)
                for vec in self.giga.embed_documents(input)]

def get_collection():
    client = chromadb.PersistentClient(path=normpath(join(dirname(__file__), '..', 'db')))
    result = []
    emb_functions = {
        "default": None,
        "Embeddings": GigaChatEmb(),
        "EmbeddingsGigaR": GigaChatEmb('EmbeddingsGigaR'),
        "SbertLarge": SentenceTransformerEmbeddingFunction(model_name='ai-forever/sbert_large_nlu_ru'),
    }
    for k, v in emb_functions.items():
        if v is None:
            coll = client.get_or_create_collection(name=k)
            v = DefaultEmbeddingFunction()
        else:
            coll = client.get_or_create_collection(name=k, embedding_function=v)
        result.append({
            "name": k,
            "coll": coll,
            "emb": v,

        })
    return result # можно попробовать завернуть в list(zip) 


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    z = get_collection()
    emb = z[0]['emb']
    print(emb(['Hello']))
    print(chromadb.PersistentClient(path=normpath(join(dirname(__file__), '..', 'db'))).list_collections())
