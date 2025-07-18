from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from os.path import normpath, join, dirname

path_to_existing_db = normpath(join(dirname(__file__), '..', '..', 'db'))
path_to_new_db = normpath(join(dirname(__file__), '..', '..', 'db'))

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

def get_from_collection(path, collection_name, embedding=embeddings):
    db = Chroma(persist_directory=path, collection_name=collection_name, embedding_function=embedding)
    documents = db.get()
    docs = [
        Document(page_content=doc)
        for doc in documents['documents']
    ]
    return docs

def add_to_collection(path, collection_name, docs, embedding=embeddings):
    db = Chroma(persist_directory=path, collection_name=collection_name, embedding_function=embedding)
    db.add_documents(docs)
    return db

documents_from_db = get_from_collection(path=path_to_existing_db, collection_name='default')
print(documents_from_db)
add_to_collection(path=path_to_new_db, collection_name='abc', docs=documents_from_db)
