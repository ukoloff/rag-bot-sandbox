import asyncio
from os import getenv

from os.path import normpath, join, dirname
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_gigachat import GigaChat, GigaChatEmbeddings

from dotenv import load_dotenv
from aiogram.filters import CommandStart
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

load_dotenv()

dp = Dispatcher()
TOKEN = getenv("BOT_TOKEN")
llm = GigaChat()

path_to_db = normpath(join(dirname(__file__), '..', '..', 'chroma.kb'))
db = Chroma(collection_name="kb.gigaRtext", embedding_function=GigaChatEmbeddings(model='Embeddings'), persist_directory=path_to_db)
retriever = db.as_retriever()

system_prompt = """Представь, что ты ассистент поддержки, отвечающий на важные
    мне вопросы исключительно на русском языке, на основании контекста, написанного ниже.
    Если не знаешь ответа, то напиши просто: "Не могу помочь с этим вопросом".
    Отвечай языком Михайлы Васильича Ломоносова.
    Контекст: {context}
    Вопрос: {question}
    Твой ответ: """

def join(docs):
    s = '\n\n'.join(doc.page_content for doc in docs)
    return s

prompt = ChatPromptTemplate.from_template(system_prompt)

chain = (
    {
        "context": retriever | join, "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    res = await llm.ainvoke(f"Привет, меня зовут {message.from_user.full_name}!")
    await message.answer(res.content)

@dp.message()
async def chat_handler(message: Message) -> None:
    res = await chain.ainvoke(message.text)
    await message.answer(res.content)

async def main() -> None:
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await dp.start_polling(bot)

if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
