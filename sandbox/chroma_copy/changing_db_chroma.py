import numpy as np
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import chromadb
from os.path import normpath, join, dirname

path_to_existing_db = normpath(join(dirname(__file__), '..', '..', 'db'))
path_to_new_db = normpath(join(dirname(__file__), '..', '..', 'new_db'))

embeddings = DefaultEmbeddingFunction()

def get_from_collection(path, collection_name, embedding=embeddings):
    client = chromadb.PersistentClient(path=path)
    db = client.get_or_create_collection(name=collection_name, embedding_function=embedding)
    documents = db.get(include=['embeddings', 'documents', 'metadatas'])
    # docs = [
    #     Document(page_content=doc, metadatas=meta)
    #     for doc, meta in zip(documents['documents'], documents['metadatas'])
    # ]
    return documents

def add_to_collection(path, collection_name, docs, embedding=embeddings):
    client = chromadb.PersistentClient(path=path)
    db = client.get_or_create_collection(name=collection_name, embedding_function=embedding)
    db.add(ids=docs['ids'], documents=docs['documents'], metadatas=docs['metadatas'], embeddings=docs['embeddings'])
    return db

documents_from_db = get_from_collection(path=path_to_existing_db, collection_name='default')
# print(documents_from_db)
add_to_collection(path=path_to_new_db, collection_name='copy_to_no_langchain', docs=documents_from_db)

client_from = chromadb.PersistentClient(path=path_to_existing_db)
client_to = chromadb.PersistentClient(path=path_to_new_db)
first_db = client_from.get_or_create_collection(name='default', embedding_function=embeddings)
second_db = client_to.get_or_create_collection(name='copy_to_no_langchain', embedding_function=embeddings)

first_embs = first_db.get(include=['embeddings'])['embeddings']
second_embs = second_db.get(include=['embeddings'])['embeddings']

print(np.allclose(first_embs[0], second_embs[0])) # проверка на то, что эмбеддинги одинаковые