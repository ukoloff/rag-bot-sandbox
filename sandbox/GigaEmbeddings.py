from gigachat import GigaChat
from dotenv import load_dotenv
from os.path import normpath, join, dirname
import chromadb

load_dotenv()
llm = GigaChat()
path_to_db = normpath(join(dirname(__file__), '..', 'db'))
db = chromadb.PersistentClient(path=path_to_db)


embeddings = llm.embeddings(texts=["Привет"], model="Embeddings")
embeddings1 = llm.embeddings(texts=["Привет"], model="EmbeddingsGigaR")

print(len(embeddings.data[0].embedding))
print(len(embeddings1.data[0].embedding))
