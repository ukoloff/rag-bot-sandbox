import asyncio
from os import getenv
import aiofiles
from os.path import normpath, join, dirname
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from langchain.schema import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from gigachat.exceptions import ResponseError
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from operator import itemgetter
from dotenv import load_dotenv
from aiogram.filters import CommandStart
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
# from load_prompt import fresh_prompt

load_dotenv()

dp = Dispatcher()
TOKEN = getenv("BOT_TOKEN")
llm = GigaChat()

path_to_db = normpath(join(dirname(__file__), "..", "..", "chroma.kb"))
path_to_file = normpath(join(dirname(__file__), "prompt.txt"))
db = Chroma(
    collection_name="kb.gigaRtext",
    embedding_function=GigaChatEmbeddings(model="Embeddings"),
    persist_directory=path_to_db,
)
retriever = db.as_retriever()
user_histories = {}


def join(docs):
    s = "\n\n".join(doc.page_content for doc in docs)
    return s


async def prompt(data):
    async with aiofiles.open(file=path_to_file, encoding="UTF-8") as f:
        text = await f.read()
    return await ChatPromptTemplate.from_messages(
        [
            (
                "system",
                text,
            ),
            # Use "chat_history" as the key for conversation history without modifying it if possible.
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "Контекст:\n{context}\nВопрос:\n{question}",
            ),  # Use user input as a variable.
        ]
    ).ainvoke(data)


chain = (
    {
        "context": itemgetter("question") | retriever | join,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | RunnableLambda(afunc=prompt, func=lambda x: x)
    | llm
    | StrOutputParser()
)


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer("Привет, я бот поддержки!")


@dp.message()
async def chat_handler(message: Message) -> None:
    async with lock:
        try:
            res = await chain_with_history.ainvoke(
                {"question": message.text},
                config=RunnableConfig(
                    configurable={"session_id": message.from_user.id}
                ),
            )
            await message.answer(res)
        except ResponseError as e:
            await message.answer(f"Ошибка GigaChat: {e.args[1]}")
        except Exception as e:
            print(e)
            await message.answer("Произошла ошибка")


store = {}


def get_session_history(session_ids):
    if session_ids not in store:  # When the session ID is not in the store.
        # Create a new ChatMessageHistory object and save it in the store.
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # Return the session history for the given session ID.


chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # A function to retrieve session history.
    input_messages_key="question",  # The key where the user's question will be inserted into the template variable.
    history_messages_key="chat_history",  # The key for the message in the history.
)


async def main() -> None:
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await dp.start_polling(bot)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    lock = asyncio.Lock()
    print("Запускаю...")
    asyncio.run(main())
