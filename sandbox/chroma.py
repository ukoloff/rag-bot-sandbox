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
system_prompt = """Представь, что ты ассистент поддержки, отвечающий на важные
    мне вопросы исключительно на русском языке, на основании определенного контекста.
    Если информации нет в контексте, то напиши кратко: "Не могу помочь с этим вопросом" и ничего больше,
    а если есть, то вставь кусок из контекста. При ответе на вопрос также ни в коем случае не упоминай, 
    что существует какой-то контекст."""

question = input("Введите ваш вопрос: ")

# 5 контекстов
first_context = db.similarity_search(query=question, k=5)[0].page_content
second_context = db.similarity_search(query=question, k=5)[1].page_content
third_context = db.similarity_search(query=question, k=5)[2].page_content
fourth_context = db.similarity_search(query=question, k=5)[3].page_content
fifth_context = db.similarity_search(query=question, k=5)[4].page_content

all_context = first_context + second_context + second_context + third_context + fourth_context + fifth_context

llm = GigaChat()

all_prompt = f"{system_prompt} \n Контекст: {all_context} \n Вопрос: {question} " + "\nТвой ответ: "

print(llm.invoke(all_prompt))