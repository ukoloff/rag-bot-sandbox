from dotenv import load_dotenv
from gigachat import GigaChat
from os.path import normpath, join, dirname
import chromadb
from sandbox.embinterface import GigaChatEmb

load_dotenv()
llm = GigaChat()
path_to_db = normpath(join(dirname(__file__), '..', '..', 'chroma.kb'))
db = chromadb.PersistentClient(path=path_to_db)
collection = db.get_or_create_collection(name="kb.gigaR", embedding_function=GigaChatEmb('EmbeddingsGigaR'))

template = """
    Представь, что ты ассистент поддержки, отвечающий на важные
    мне вопросы исключительно на русском языке, на основании контекста, написанного ниже.
    Если информации нет в контексте, то напиши кратко: "Не могу помочь с этим вопросом" и ничего больше.
    При ответе на вопрос также ни в коем случае не упоминай, 
    что существует какой-то контекст. Отвечай языком Михаила Васильевича Ломоносова.
"""

question = input("Введите ваш вопрос: ")
docs = collection.query(query_texts=[question], n_results=5)
all_context = '\n\n'.join(docs['documents'][0])
all_prompt = f"{template} \n Контекст: {all_context} \n Вопрос: {question} " + "\nТвой ответ: "
print(llm.chat(all_prompt).choices[0].message.content)