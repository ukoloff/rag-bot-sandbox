from dotenv import load_dotenv
from os.path import normpath, join, dirname
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_gigachat import GigaChat, GigaChatEmbeddings

load_dotenv()
llm = GigaChat()
path_to_db = normpath(join(dirname(__file__), '..', '..', 'chroma.kb'))
db = Chroma(collection_name="kb.gigaR", embedding_function=GigaChatEmbeddings(model='EmbeddingsGigaR'), persist_directory=path_to_db)
retriever = db.as_retriever()

system_prompt = """Представь, что ты ассистент поддержки, отвечающий на важные
    мне вопросы исключительно на русском языке, на основании контекста, написанного ниже.
    Если информации нет в контексте, то напиши кратко: "Не могу помочь с этим вопросом" и ничего больше.
    При ответе на вопрос также ни в коем случае не упоминай, 
    что существует какой-то контекст. Отвечай языком Михаила Васильевича Ломоносова.
    Контекст: {context}
    Вопрос: {question}
    Твой ответ:"""

def join(docs):
    s = '\n\n'.join(doc.page_content for doc in docs)
    return s

prompt = ChatPromptTemplate.from_template(system_prompt)
question = input("Введите ваш вопрос: ")

def view(x):
    return x

chain = (
    {
        "context": retriever | join, "question": RunnablePassthrough()
    }
    | RunnableLambda(view)
    | prompt
    | llm
)
print(chain.invoke(question).content)