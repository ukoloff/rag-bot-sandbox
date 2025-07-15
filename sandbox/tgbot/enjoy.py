import asyncio
from os import getenv

from os.path import normpath, join, dirname
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from langchain.schema import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from gigachat.exceptions import ResponseError

from dotenv import load_dotenv
from aiogram.filters import CommandStart
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from load_prompt import fresh_prompt

load_dotenv()

dp = Dispatcher()
TOKEN = getenv("BOT_TOKEN")
llm = GigaChat()

path_to_db = normpath(join(dirname(__file__), '..', '..', 'chroma.kb'))
db = Chroma(collection_name="kb.gigaRtext", embedding_function=GigaChatEmbeddings(model='Embeddings'), persist_directory=path_to_db)
retriever = db.as_retriever()
user_histories = {}

def join(docs):
    s = '\n\n'.join(doc.page_content for doc in docs)
    return s

chain = (
    {
        "context": retriever | join, "question": RunnablePassthrough()
    }
    | fresh_prompt
    # | add_hi
    | llm
    # | store_hi
)

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    async with lock:
        try:
            user_id = message.from_user.id
            if user_id not in user_histories:
                user_histories[user_id] = InMemoryChatMessageHistory()
            history = user_histories[user_id]
            text = f"Привет, меня зовут {message.from_user.full_name}!"
            history.add_message(HumanMessage(content=text))
            res = await llm.ainvoke(text)
            history.add_message(AIMessage(content=res.content))
            await message.answer(res.content)
        except ResponseError as e:
            await message.answer(f"Ошибка GigaChat: {e.args[1]}")
        except Exception as e:
            await message.answer("Произошла ошибка")

@dp.message()
async def chat_handler(message: Message) -> None:
    async with lock:
        try:
            user_id = message.from_user.id
            if user_id not in user_histories:
                user_histories[user_id] = InMemoryChatMessageHistory()
            history = user_histories[user_id]
            text = message.text
            history.add_message(HumanMessage(content=text))
            res = await chain.ainvoke(text)
            history.add_message(AIMessage(content=res.content))
            await message.answer(res.content)
        except ResponseError as e:
            await message.answer(f"Ошибка GigaChat: {e.args[1]}")
        except Exception as e:
            print(e)
            await message.answer("Произошла ошибка")

async def main() -> None:
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await dp.start_polling(bot)

if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    lock = asyncio.Lock()
    print("Запускаю...")
    asyncio.run(main())
