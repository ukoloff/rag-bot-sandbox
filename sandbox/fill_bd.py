from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from os.path import normpath, join, dirname
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embinterface import get_collection

load_dotenv()

path_to_txt = normpath(join(dirname(__file__), 'мастер_и_маргарита.txt'))
loader = TextLoader(path_to_txt, encoding="UTF-8")
document = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
texts = text_splitter.split_documents(document)
new_texts = [doc.page_content for doc in texts]

for item in get_collection():
    coll = item.coll
    if len(coll.get(limit=1)['ids']) > 0:
        continue
    print("Заполняю коллекцию")
    coll.add(
        ids=[f"{i}" for i in range(len(new_texts))],
        documents=new_texts
    )