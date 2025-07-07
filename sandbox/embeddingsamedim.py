# Проверка на то, что при одинаковой размерности эмбеддингов (при этом, у них разные Embedding Function), поиск похожего контекста будет производиться неверно и будет выдаваться мусор
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from os.path import normpath, join, dirname
import chromadb

path_to_db = normpath(join(dirname(__file__), '..', 'db'))
db = chromadb.PersistentClient(path=path_to_db)
names = ["BAAI/bge-small-en-v1.5", "snowflake/snowflake-arctic-embed-xs", "snowflake/snowflake-arctic-embed-s", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
collections = []
collections.append(db.get_or_create_collection(name="abc"))
for name in names:
    collections.append(db.get_or_create_collection(name="abc", embedding_function=SentenceTransformerEmbeddingFunction(model_name=name)))

question = input("Введите ваш вопрос: ")

for collection in collections:
    docs = collection.query(query_texts=[question], n_results=5)
    all_context = '\n\n'.join(docs['documents'][0])
    print(all_context)
    print("___________________________")