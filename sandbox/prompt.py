import asyncio

import aiofiles
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from os.path import normpath, join, dirname

path_to_file = normpath(join(dirname(__file__), "prompt.txt"))
system_prompt = """
    Контекст: {context}
    Вопрос: {question}
    Твой ответ: """

input = {"context": "Контекст", "question": "Вопрос"}

def my_prompt(data):
    with open(file=path_to_file, encoding="UTF-8") as f:
        text = f.read()
    return ChatPromptTemplate.from_template(text + system_prompt).invoke(data)

async def amy_prompt(data):
    async with aiofiles.open(file=path_to_file, encoding="UTF-8") as f:
        text = await f.read()
    return await ChatPromptTemplate.from_template(text + system_prompt).ainvoke(data)

p2 = RunnableLambda(my_prompt, amy_prompt)
a2 = p2.invoke(input)
print(a2)

async def main():
    aa = await p2.ainvoke(input)
    print(aa)

asyncio.run(main())
