import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_gigachat import GigaChatEmbeddings
from langchain_chroma import Chroma
from os.path import normpath, join, dirname
from dotenv import load_dotenv

load_dotenv()

pdf_folder_path = normpath(join(dirname(__file__), '..', 'pdfs'))
db_path = normpath(join(dirname(__file__), '..', 'chroma.kb'))

def load_document(pdf_folder_path):
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            file_path = join(pdf_folder_path, file)
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(e)
    return documents

def split_text(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    return texts

def update_vector_db(texts, persist_dir, collection_name):
    embeddings = GigaChatEmbeddings()
    db = Chroma(persist_directory=persist_dir, collection_name=collection_name, embedding_function=embeddings)
    db.add_documents(texts)
    return db

documents = load_document(pdf_folder_path)
chunks = split_text(documents)
db = update_vector_db(chunks, db_path, 'kb.gigaRtext')