from dotenv import load_dotenv
from gigachat import GigaChat
from os.path import normpath, join, dirname
import chromadb
from sandbox.embinterface import GigaChatEmb
from gigachat.models import Chat, Messages, MessagesRole

from sandbox.translate import template

load_dotenv()
llm = GigaChat()
path_to_db = normpath(join(dirname(__file__), '..', '..', 'db'))
db = chromadb.PersistentClient(path=path_to_db)
# print(db.list_collections())
collection = db.get_or_create_collection(name="EmbeddingsGigaR", embedding_function=GigaChatEmb('EmbeddingsGigaR'))
# question = input("Вопрос: ")
# print(llm.embeddings(texts=['hi!']).data[0])
# print(collection.query(query_texts='hi!', n_results=3))

template = """
    Представь, что ты ассистент поддержки, отвечающий на важные
    мне вопросы исключительно на русском языке, на основании определенного контекста.
    Если информации нет в контексте, то напиши кратко: "Не могу помочь с этим вопросом" и ничего больше,
    а если есть, то вставь кусок из контекста. При ответе на вопрос также ни в коем случае не упоминай, 
    что существует какой-то контекст.
"""

messages = [
    Messages(
        role=MessagesRole.SYSTEM,
        content = template
    ),
]

while True:
    system_prompt = """Представь, что ты ассистент поддержки, отвечающий на важные
        мне вопросы исключительно на русском языке, на основании определенного контекста.
        Если информации нет в контексте, то напиши кратко: "Не могу помочь с этим вопросом" и ничего больше,
        а если есть, то вставь кусок из контекста. При ответе на вопрос также ни в коем случае не упоминай, 
        что существует какой-то контекст.
        Контекст: {context}
        Вопрос: {question}
        Твой ответ:"""
    prompt = ChatPromptTemplate.from_template(system_prompt)

    return (
            {
                "context": db.as_retriever(), "question": RunnablePassthrough()
            }
            | prompt
            | llm
        # | StrOutputParser
    )