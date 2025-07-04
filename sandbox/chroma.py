from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
#import chromadb # На будущее
from langchain_gigachat import GigaChat
from dotenv import load_dotenv
from os.path import normpath, join, dirname

load_dotenv()

path_to_db = normpath(join(dirname(__file__), '..', 'db'))

## Подгрузка БД Chroma
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=path_to_db, collection_name="abc", embedding_function=embeddings)

print(db.similarity_search(query="Какие фестивали пройдут 5 июля в Екатеринбурге?", k=3)) # возврат массива с похлжими эмбеддингами

llm = GigaChat()
question = "Какие фестивали пройдут 5 июля в Екатеринбурге??"

print(llm.invoke(question))