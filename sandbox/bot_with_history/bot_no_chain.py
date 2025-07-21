import asyncio

from os import getenv
from dotenv import load_dotenv
from os.path import normpath, join, dirname

from langchain_chroma import Chroma
# from langchain_core.prompts import ChatPromptTemplate
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from gigachat.exceptions import ResponseError
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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
path_to_db = normpath(join(dirname(__file__), '..', '..', 'chroma.kb'))
db = Chroma(collection_name="kb.gigaRtext", embedding_function=GigaChatEmbeddings(model='Embeddings'), persist_directory=path_to_db)
lock = asyncio.Lock()

system_prompt = """Представь, что ты ассистент поддержки, отвечающий на важные
    мне вопросы исключительно на русском языке, на основании контекста, написанного ниже.
    Если информации нет в контексте, то напиши кратко: "Не могу помочь с этим вопросом" и ничего больше.
    При ответе на вопрос также ни в коем случае не упоминай, 
    что существует какой-то контекст. Отвечай языком Михаила Васильевича Ломоносова.
    """

def join(docs):
    s = '\n\n'.join(doc.page_content for doc in docs)
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
    this_history = history[id]
    async with lock:
        try:
            question = message.text
            context = join(db.similarity_search(query=str(question), k=5))
            prompt = f"""{system_prompt}
Контекст: {context}
Вопрос: {question}
Твой ответ: """
            this_history.append(HumanMessage(content=prompt))
            answer_llm = await llm.ainvoke(this_history)
            this_history.append(AIMessage(content=answer_llm.content))
            await message.answer(answer_llm.content, parse_mode=ParseMode.MARKDOWN)
        except ResponseError as e:
            await message.answer(f"Ошибка GigaChat: {e.args[1]}")
        except Exception as e:
            print(e)
            await message.answer("Произошла ошибка")

# prompt = ChatPromptTemplate.from_template(system_prompt)
# question = input("Введите ваш вопрос: ")
#
# context = join(db.similarity_search(query=question, k=5))
#
# answer = llm.invoke(system_prompt + '\n' + "Контекст: " + context + '\n' +
#                     'Вопрос: ' + question + '\n' + "Твой ответ: ")

# print(answer.content)

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