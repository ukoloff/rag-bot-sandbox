# Проверка на то, что при одинаковой размерности эмбеддингов (при этом, у них разные Embedding Function), поиск похожего контекста будет производиться неверно и будет выдаваться мусор
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from os.path import normpath, join, dirname
import chromadb

path_to_db = normpath(join(dirname(__file__), '..', 'db'))
db = chromadb.PersistentClient(path=path_to_db)
names = ["BAAI/bge-small-en-v1.5", "snowflake/snowflake-arctic-embed-xs", "snowflake/snowflake-arctic-embed-s", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
collections = []
collections.append(db.get_or_create_collection(name="abc"))
for name in names:
    collections.append(db.get_or_create_collection(name="abc", embedding_function=SentenceTransformerEmbeddingFunction(model_name=name)))

question = input("Введите ваш вопрос: ")

for collection in collections:
    docs = collection.query(query_texts=[question], n_results=5)
    all_context = '\n\n'.join(docs['documents'][0])
    print(all_context)
    print("___________________________")

path_to_html = normpath(join(dirname(__file__), 'output', 'a.html'))
f = open(path_to_html, 'w', encoding='utf-8')
f.write(f"""
<meta charset="utf-8">
<h1><b>Вопрос</b>: {question}</h1>
<hr>
""")
n_result = 5
for collection in collections:
    docs = collection.query(query_texts=[question], n_results=n_result)
    f.write("<table>")
    for n in range(1, n_result+1):
        f.write(f"""
    <tr>
        <td>
            <h2>
                <b><pre><u>Ответ {n}:</u>   </pre></b>
            </h2>
        </td>
        <td><h2>
            <b><pre>{docs['documents'][0][n-1]}</pre></b>
        </h2></td>
    </tr>
""")
    f.write("""
</table>
<hr>""")