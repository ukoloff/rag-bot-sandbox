import asyncio

from html import escape
from os import getenv
from dotenv import load_dotenv
from os.path import normpath, join, dirname

from langchain_chroma import Chroma
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from sentence_transformers import CrossEncoder
from gigachat.exceptions import ResponseError
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import aiofiles
from aiofiles.ospath import exists
from aiogram.client.default import DefaultBotProperties
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart

load_dotenv()
TOKEN = getenv("BOT_TOKEN")
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
dp = Dispatcher()
llm = GigaChat()
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
path_to_db = normpath(join(dirname(__file__), '..', '..', 'chroma.kb'))
db = Chroma(collection_name="kb.gigaRtext", embedding_function=GigaChatEmbeddings(model='Embeddings'), persist_directory=path_to_db)
lock = asyncio.Lock()
path_to_log = normpath(join(dirname(__file__), '..', 'output'))

top_k = 5  # Столько результатов извлекается после ReRank

system_prompt = """Представь, что ты ассистент поддержки, отвечающий на важные
    мне вопросы исключительно на русском языке, на основании контекста, написанного ниже.
    Если информации нет в контексте, то напиши кратко: "Не могу помочь с этим вопросом" и ничего больше.
    При ответе на вопрос также ни в коем случае не упоминай, 
    что существует какой-то контекст. Отвечай языком Михаила Васильевича Ломоносова.
    """

def escape_html(str):
    return escape(str).replace("\n", "<br>")

async def write_start_html(path):
    async with aiofiles.open(path, 'a', encoding='utf-8') as f:
        await f.write(f"""<html>
<head>
<meta charset="utf-8">
<style>
details > details {{
	border-left: 1px solid navy;
  padding-left: 1em;
}}
</style>
</head>
<body>
""")

async def write_html(path, question, answer, context, scores):
    async with aiofiles.open(path, 'a', encoding='utf-8') as f:
        await f.write(f"""<h1><b>Вопрос</b>: {question}</h1>
<hr>
<details>
    <summary>Ответ LLM: </summary>
    <p>{escape_html(answer)}</p>
</details>
<details>
    <summary>Чанки: </summary>
""")
        for i, con in enumerate(context):
            await f.write(f"""    <details>
        <summary>Чанк {i+1} (score: {scores[i]}): </summary>
        <p>{escape_html(con.page_content)}</p>>
    </details>
""")
        await f.write("</details>\n")

def join_context(docs):
    s = "\n\n".join(doc for doc in docs)
    return s

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer("Привет, я бот поддержки!")
history = {}

@dp.message()
async def chat_handler(message: Message) -> None:
    id = message.chat.id
    if id not in history:
        history[id] = []
    path_to_html = join(path_to_log, f"{id}.html")
    if not await exists(path_to_html):
        await write_start_html(path_to_html)
    this_history = history[id]
    if len(this_history) >= 20:
        this_history = this_history[2:]
    async with lock:
        try:
            question = message.text
            context = db.similarity_search(query=question, k=50)
            pairs = [[question, doc.page_content] for doc in context]
            scores = rerank_model.predict(pairs)
            ranked_docs = sorted(zip(context, scores), reverse=True, key=lambda x: x[1])
            context_list = [doc.page_content for doc, score in ranked_docs[:top_k]]  # top_k элементов
            scores_list = [score for doc, score in ranked_docs[:top_k]]
            context_joined_to_str = join_context(context_list)
            prompt = f"""Контекст: {context_joined_to_str}
Вопрос: {question}
Твой ответ: """
            this_history.append(HumanMessage(content=prompt))
            answer_llm = await llm.ainvoke([SystemMessage(system_prompt)] + this_history)
            await write_html(path_to_html, question, answer_llm.content, context[:5], scores_list)
            this_history.append(AIMessage(content=answer_llm.content))
            await message.answer(answer_llm.content, parse_mode=ParseMode.MARKDOWN)
        except ResponseError as e:
            await message.answer(f"Ошибка GigaChat: {e.args[1]}")
        except Exception as e:
            print(e)
            await message.answer("Произошла ошибка")

async def main() -> None:
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
    print('...')

def start_bot():
    print("Запускаю...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот завершил работу.")

start_bot()