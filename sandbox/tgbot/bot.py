import asyncio
from os import getenv

from dotenv import load_dotenv
from aiogram.filters import CommandStart
from aiogram import Bot, Dispatcher, html
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from langchain_gigachat import GigaChat, GigaChatEmbeddings

load_dotenv()

dp = Dispatcher()
TOKEN = getenv("BOT_TOKEN")
llm = GigaChat()

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    res = await llm.ainvoke(f"Привет, меня зовут {message.from_user.full_name}!")
    await message.answer(res.content)

@dp.message()
async def echo_handler(message: Message) -> None:
    try:
        await message.send_copy(chat_id=message.chat.id)
    except TypeError:
        await message.answer("Nice try!")

async def main() -> None:
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await dp.start_polling(bot)

if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
    