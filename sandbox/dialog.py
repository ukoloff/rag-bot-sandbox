import chromadb
from  gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from dotenv import load_dotenv
from os.path import normpath, join, dirname

load_dotenv()
llm = GigaChat()
path_to_db = normpath(join(dirname(__file__), '..', 'db'))
db = chromadb.PersistentClient(path=path_to_db)

collection = db.get_or_create_collection(name="abc")

system_prompt = """Представь, что ты ассистент поддержки, отвечающий на важные
    мне вопросы исключительно на русском языке, на основании определенного контекста.
    """
messages = [
    Messages(
        role=MessagesRole.SYSTEM,
        content = system_prompt
    )]

while True:
    question = input("Введите ваш вопрос: ")
    if question == "":
        break
    docs = collection.query(query_texts=[question], n_results=5)
    all_context = '\n\n'.join(docs['documents'][0])

    messages.append(Messages(
        role=MessagesRole.USER,
        content = f"Контекст: {all_context} \n Вопрос: {question} " + "\nТвой ответ: "))

    call_gigachat = llm.chat(Chat(messages=messages))
    print(call_gigachat.choices[0].message.content)
    messages.append(call_gigachat.choices[0].message)