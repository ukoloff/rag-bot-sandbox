#from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_chroma import Chroma
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import chromadb # На будущее
from langchain_gigachat import GigaChat
from dotenv import load_dotenv
from os.path import normpath, join, dirname

load_dotenv()
llm = GigaChat(model="GigaChat-Max")
path_to_db = normpath(join(dirname(__file__), '..', 'db'))

## Подгрузка БД Chroma
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path=path_to_db)

collection = db.get_or_create_collection(name="abc")

# db = Chroma(persist_directory=path_to_db, collection_name="abc", embedding_function=DefaultEmbeddingFunction)
system_prompt = """Представь, что ты ассистент поддержки, отвечающий на важные
    мне вопросы исключительно на русском языке, на основании определенного контекста.
    Если информации нет в контексте, то напиши кратко: "Не могу помочь с этим вопросом" и ничего больше,
    а если есть, то вставь кусок из контекста. При ответе на вопрос также ни в коем случае не упоминай, 
    что существует какой-то контекст.
    Отвечай стихами"""

question = input("Введите ваш вопрос: ")

# 5 контекстов
docs = collection.query(query_texts=[question], n_results=5)
all_context = '\n\n'.join(docs['documents'][0])

all_prompt = f"{system_prompt} \n Контекст: {all_context} \n Вопрос: {question} " + "\nТвой ответ: "
call_gigachat = llm.invoke(all_prompt)

print(call_gigachat.content)