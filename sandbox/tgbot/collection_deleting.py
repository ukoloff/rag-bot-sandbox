import chromadb
from os.path import normpath, join, dirname
import os

pdf_folder_path = normpath(join(dirname(__file__), '..', '..', 'pdfs'))

db_path = normpath(join(dirname(__file__), '..', '..', 'chroma.kb'))
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name='kb.gigaR')
print(collection.get(include=['metadatas']))

# for file in os.listdir(pdf_folder_path):
#     if file.endswith('.pdf'):
#         file_path = join(pdf_folder_path, file)
#         print(file_path)
#         collection.delete(where={'source': file_path})