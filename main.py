from dotenv import load_dotenv
from gigachat import GigaChat as GiChat

question = "Какой плащ был у Понтия Пилата?"

load_dotenv()

llmG = GiChat()

resG = llmG.chat(question)

print(resG.choices[0].message.content)
print(resL.content)
