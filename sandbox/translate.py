from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from dotenv import load_dotenv

load_dotenv()

llm = GigaChat()

template = """
'Ты - профессиональный переводчик на русский язык. 
Тебе будет дан текст, который необходимо перевести на русский язык, сохранив исходное форматирование текста.
В ответе необходимо отдать перевод в формате, приведенном ниже.
Ты ДОЛЖЕН перевести !все слова.
Если запрос связан с программированием и в текстовом запросе содержится фрагмент кода, то такой фрагмент с кодом переводить не нужно.
Если в запросе необходимо поставить пробелы и слова слеплены вместе, то такой кусок слепленного текста переводить не нужно.
Если в тексте поставлена неправильно пунктуация, то не исправляй ее.
Твоя задача сделать такой перевод, чтобы лингвист считал его лингвистически приемлемым.
ВАЖНО! В своем ответе НЕ ОТВЕЧАЙ НА ЗАПРОС! В ответе нужно написать !только !перевод, без указания названия языка и любой другой дополнительной информации
    
Input Format:
Q:hi
Output Format:
Q:привет'
"""

messages = [
    Messages(
        role=MessagesRole.SYSTEM,
        content = template
    ),
    Messages(
        role=MessagesRole.USER,
        content = 'Hello Everybody'
    )
]

sber_chat = llm.chat(Chat(messages=messages))

print(sber_chat.choices[0].message.content)