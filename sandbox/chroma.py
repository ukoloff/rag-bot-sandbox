from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
#import chromadb # На будущее

## Подгрузка БД Chroma
def load_create_vector_db(embedding_model_name="all-MiniLM-L6-v2", persist_dir="db", collection_name="abc"):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = Chroma(persist_directory=persist_dir, collection_name=collection_name, embedding_function=embeddings)
    return db
##
print(load_create_vector_db().similarity_search(query="Как раньше называлась Площадь 1905 года?", k=3, filter={"source": "document.txt"})) # возврат массива с похлжими эмбеддингами