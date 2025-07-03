from dotenv import load_dotenv
from gigachat import GigaChat as GiChat
load_dotenv()

question = "Какой плащ был у Понтия Пилата?"

llmG = GiChat()

resG = llmG.chat(question)

print(resG.choices[0].message.content)
