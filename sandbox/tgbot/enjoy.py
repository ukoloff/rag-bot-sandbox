import asyncio
from os import getenv
import aiofiles
from aiofiles.ospath import exists
from os.path import normpath, join, dirname

from html import escape

from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda
from langchain_gigachat import GigaChat, GigaChatEmbeddings
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

load_dotenv()

TOKEN = getenv("BOT_TOKEN")
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
llm = GigaChat()

path_to_db = normpath(join(dirname(__file__), "..", "..", "chroma.kb"))
path_to_file = normpath(join(dirname(__file__), "prompt.txt"))
path_to_log = normpath(join(dirname(__file__), '..', 'output'))
db = Chroma(
    collection_name="kb.gigaRtext",
    embedding_function=GigaChatEmbeddings(model="Embeddings"),
    persist_directory=path_to_db,
)
retriever = db.as_retriever(search_kwargs={"k": 5})
user_histories = {}

def escape_html(str):
    return escape(str).replace("\n", "<br>")

async def write_start_html(path):
    f = await aiofiles.open(path, 'w', encoding='utf-8')
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

async def write_html(path, question, answer, context):
    f = await aiofiles.open(path, 'a', encoding='utf-8')
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
        <summary>Чанк {i+1}: </summary>
        <p>{escape_html(con)}</p>>
    </details>
""")
    await f.write("</details>\n")

context_list = []

def join_context(docs):
    s = "\n\n".join(doc.page_content for doc in docs)
    context_list.clear()
    context_list.extend(doc.page_content for doc in docs)
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
        "context": itemgetter("question") | retriever | join_context,
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
    id = message.chat.id
    path_to_html = join(path_to_log, f"{id}.html")
    if not await exists(path_to_html):
        await write_start_html(path_to_html)
    async with lock:
        try:
            question = message.text
            res = await chain_with_history.ainvoke(
                {"question": question},
                config=RunnableConfig(
                    configurable={"session_id": message.from_user.id}
                ),
            )
            await write_html(path=path_to_html, question=question, answer=res, context=context_list)
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
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
    print('...')

lock = asyncio.Lock()

def start_bot():
    # logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    print("Запускаю...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот завершил работу.")