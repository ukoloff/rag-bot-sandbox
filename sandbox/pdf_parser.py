import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_gigachat import GigaChatEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from os.path import normpath, join, dirname, exists
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

def split_text(documents, chunk_size=3000, chunk_overlap=200, min_chunk_size=300):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)

    filtered_texts = []
    i = 0
    while i < len(texts):
        current_text = texts[i]
        if len(current_text.page_content) < min_chunk_size and i < len(texts) - 1:
            merged_content = current_text.page_content + " " + texts[i + 1].page_content
            merged_metadata = {**current_text.metadata, **texts[i + 1].metadata}
            merged_doc = Document(page_content=merged_content, metadata=merged_metadata)
            filtered_texts.append(merged_doc)
            i += 2
        else:
            filtered_texts.append(current_text)
            i += 1
    return filtered_texts

def update_vector_db(texts, persist_dir, collection_name):
    embeddings = GigaChatEmbeddings(model='EmbeddingsGigaR')
    db = Chroma(persist_directory=persist_dir, collection_name=collection_name, embedding_function=embeddings)
    db.add_documents(texts)
    return db

documents = load_document(pdf_folder_path)
chunks = split_text(documents)
db = update_vector_db(chunks, db_path, 'kb.gigaR')