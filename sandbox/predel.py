from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from langchain_gigachat import GigaChat
from dotenv import load_dotenv
from os.path import normpath, join, dirname
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
llm = GigaChat()

path_to_txt = normpath(join(dirname(__file__), 'война_и_мир.txt'))

path_to_write = normpath(join(dirname(__file__), 'output', 'война_и_мир_output.txt'))

def load_document(file_path):
    loader = TextLoader(file_path, encoding="windows-1251")
    documents = loader.load()
    return documents

doc = load_document(path_to_txt)
print(doc[0])

prompt = ChatPromptTemplate.from_messages(
    [("system", "Напишите краткое резюме следующего текста:\\n\\n{context}")]
)

# Создание цепочки
chain = create_stuff_documents_chain(llm, prompt)

# Вызов цепочки
result = chain.invoke({"context": load_document(path_to_txt)})

with open(path_to_write, 'w', encoding='UTF-8') as f:
    f.write(result)


# print(len(load_document(path_to_txt)[0].page_content))