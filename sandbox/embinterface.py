from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
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
        result = []
        for vec in self.giga.embed_documents(input):
            result.append(np.array(vec))
        return result

def get_collection():
    client = chromadb.PersistentClient(path=normpath(join(dirname(__file__), '..', 'db')))
    result = []
    result.append(client.get_or_create_collection(name="default"))
    result.append(client.get_or_create_collection(name="Embeddings",
                                                  embedding_function=GigaChatEmb()))
    result.append(client.get_or_create_collection(name="EmbeddingsGigaR",
                                                  embedding_function=GigaChatEmb('EmbeddingsGigaR')))
    result.append(client.get_or_create_collection(name="SbertLarge",
                                                  embedding_function=SentenceTransformerEmbeddingFunction(model_name='ai-forever/sbert_large_nlu_ru')))
    return result