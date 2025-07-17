from langchain_chroma import Chroma
from os.path import normpath, join, dirname

path_to_existing_db = normpath(join(dirname(__file__), '..', '..', 'db'))
path_to_new_db = normpath(join(dirname(__file__), '..', '..', 'db'))

def get_from_collection(path, collection_name):
    db = Chroma(persist_directory=path, collection_name=collection_name)
    documents = db.get()
    return documents

def add_to_collection(path, collection_name, docs):
    db = Chroma(persist_directory=path, collection_name=collection_name)
    documents = db.add_documents(docs)
    return db

documents = get_from_collection(path=path_to_existing_db, collection_name='default')
print(add_to_collection(path=path_to_new_db, collection_name='abc', docs=documents))
