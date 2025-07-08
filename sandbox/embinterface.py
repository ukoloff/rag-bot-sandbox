from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_gigachat import GigaChatEmbeddings
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# giga = GigaChatEmbeddings(model='Embeddings')
# sbert = SentenceTransformerEmbeddingFunction(model_name='ai-forever/sbert_large_nlu_ru')

# print("end")

# print('GigaChat', giga.embed_query('Hi!'))
# print('X', sbert(['Hi!']))

class GigaChatEmb(EmbeddingFunction):

    def __init__(self, model_name='Embeddings'):
        super().__init__()
        self.giga = GigaChatEmbeddings(model=model_name)

    def __call__(self, input: Documents) -> Embeddings:
        result = []
        for vec in self.giga.embed_documents(input):
            result.append(np.array(vec))
        return result

test = GigaChatEmb()
print('Test', test(['Hi!']))

testR = GigaChatEmb('EmbeddingsGigaR')
print('TestR', testR(['Hi!']))