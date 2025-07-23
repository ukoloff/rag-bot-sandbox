from os.path import normpath, join, dirname
from langchain_chroma import Chroma
from langchain_gigachat import GigaChatEmbeddings
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

path_to_db = normpath(join(dirname(__file__), '..', '..', 'chroma.kb'))
db = Chroma(collection_name="kb.gigaRtext", embedding_function=GigaChatEmbeddings(model='Embeddings'), persist_directory=path_to_db)

question = input("Введите вопрос: ")
context = db.similarity_search(query=question, k=50)

pairs = [[question, doc.page_content] for doc in context]
scores = model.predict(pairs)
ranked_docs = sorted(zip(context, scores), reverse=True, key=lambda x: x[1])

top_k = 5
for doc, score in ranked_docs[:top_k]:
    print(f"""Score: {score} | content: {doc.page_content}
---------------------------------------------""")