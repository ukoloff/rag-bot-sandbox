from os.path import normpath, join, dirname
from langchain_chroma import Chroma
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from dotenv import load_dotenv

load_dotenv()

path_to_db = normpath(join(dirname(__file__), '..', '..', 'chroma.kb'))
db = Chroma(collection_name="kb.gigaRtext", embedding_function=GigaChatEmbeddings(model='Embeddings'), persist_directory=path_to_db)

def join_context(docs):
    s = '\n\n'.join(doc.page_content for doc in docs)
    return s

question = input()
context = db.similarity_search(query=question, k=5)
all_context = join_context(context)
print(all_context)
