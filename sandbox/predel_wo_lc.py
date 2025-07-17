from langchain_community.document_loaders import TextLoader
from gigachat import GigaChat
from dotenv import load_dotenv
from os.path import normpath, join, dirname

load_dotenv()
llm = GigaChat()

path_to_txt = normpath(join(dirname(__file__), 'война_и_мир.txt'))

path_to_write = normpath(join(dirname(__file__), 'output', 'война_и_мир_output.txt'))

def load_document(file_path):
    loader = TextLoader(file_path, encoding="windows-1251")
    documents = loader.load()
    return documents

start = 0
finish = 1000

context = load_document(path_to_txt)[0].page_content

file = open(path_to_write, 'w', encoding='UTF-8')

while finish < len(context):
    fragment = context[start:finish]
    start += 1000
    finish += 2000
    all_prompt = f"Напишите краткое резюме следующего текста:\n\n{fragment}"
    chatting = llm.chat(all_prompt)
    print(chatting.usage)
    file.write(chatting.choices[0].message.content + '\n\n --- \n\n')
# print(llm.chat(all_prompt).choices[0].message.content)
#



# print(len(load_document(path_to_txt)[0].page_content))