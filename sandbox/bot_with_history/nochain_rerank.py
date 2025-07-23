import asyncio

from os import getenv
import time
from dotenv import load_dotenv
from os.path import normpath, join, dirname

from langchain_chroma import Chroma
# from langchain_core.prompts import ChatPromptTemplate
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from sentence_transformers import CrossEncoder
from gigachat.exceptions import ResponseError
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import aiofiles
from aiogram.client.default import DefaultBotProperties
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart

load_dotenv()
TOKEN = getenv("BOT_TOKEN")
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
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

def join_context(docs):
    s = '\n\n'.join(doc for doc in docs)
    return s

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer("Привет, я бот поддержки!")
# id user, name user, в txt, utf-8
# парс в output: время, сообщение пользователя, время, ответ ИИ
history = {}

@dp.message()
async def chat_handler(message: Message) -> None:
    id = message.chat.id
    if id not in history:
        history[id] = []
    file_path = join(path_to_log, f"{id}.txt")
    f = await aiofiles.open(file=file_path, encoding="UTF-8", mode='a')
    await f.write(f"""Логин пользователя: {message.from_user.full_name}
Username: {message.from_user.username}\n""")
    this_history = history[id]
    if len(this_history) >= 20:
        this_history = this_history[2:]
    async with lock:
        try:
            question = message.text
            await f.write(f"""Время запроса: {time.ctime(time.time())}
Ввод пользователя: {question}\n""")
            context = db.similarity_search(query=question, k=50)
            pairs = [[question, doc.page_content] for doc in context]
            scores = rerank_model.predict(pairs)
            ranked_docs = sorted(zip(context, scores), reverse=True, key=lambda x: x[1])
            context_list = [doc.page_content for doc, score in ranked_docs[:top_k]] # top_k элементов
            context_joined_to_str = join_context(context_list)
            prompt = f"""Контекст: {context_joined_to_str}
Длина символов контекста: {len(context_joined_to_str)}
Вопрос: {question}
Твой ответ: """
            this_history.append(HumanMessage(content=prompt))
            answer_llm = await llm.ainvoke([SystemMessage(system_prompt)] + this_history)
            this_history.append(AIMessage(content=answer_llm.content))
            await f.write(f"""Время ответа от LLM: {time.ctime(time.time())}
Ответ LLM: {answer_llm.content}\n""")
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
    # logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    print("Запускаю...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот завершил работу.")

start_bot()