import numpy as np
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from os.path import normpath, join, dirname

path_to_existing_db = normpath(join(dirname(__file__), '..', '..', 'db'))
path_to_new_db = normpath(join(dirname(__file__), '..', '..', 'db'))

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

def get_from_collection(path, collection_name, embedding=embeddings):
    db = Chroma(persist_directory=path, collection_name=collection_name, embedding_function=embedding)
    documents = db.get(include=['embeddings', 'documents', 'metadatas'])
    # docs = [
    #     Document(page_content=doc, metadatas=meta)
    #     for doc, meta in zip(documents['documents'], documents['metadatas'])
    # ]
    return documents

def add_to_collection(path, collection_name, docs, embedding=embeddings):
    db = Chroma(persist_directory=path, collection_name=collection_name, embedding_function=embedding)
    db._collection.add(ids=docs['ids'],
        embeddings=docs['embeddings'], metadatas=docs['metadatas'], documents=docs['documents'])
    return db

documents_from_db = get_from_collection(path=path_to_existing_db, collection_name='default')
# print(documents_from_db)
add_to_collection(path=path_to_new_db, collection_name='copy_to', docs=documents_from_db)

# first_db = Chroma(persist_directory=path_to_existing_db, collection_name='default', embedding_function=embeddings)
# second_db = Chroma(persist_directory=path_to_new_db, collection_name='copy_to', embedding_function=embeddings)
#
# first_embs = first_db.get(include=['embeddings'])['embeddings']
# second_embs = second_db.get(include=['embeddings'])['embeddings']
#
# print(np.allclose(first_embs[0], second_embs[0])) # проверка на то, что эмбеддинги одинаковые