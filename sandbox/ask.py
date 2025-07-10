from embinterface import get_collection
from os.path import normpath, join, dirname
from dotenv import load_dotenv

load_dotenv()

question = input("Введите вопрос: ")

path_to_html = normpath(join(dirname(__file__), 'output', 'a.html'))
f = open(path_to_html, 'w', encoding='utf-8')
f.write(f"""
<html>
<head>
<meta charset="utf-8">
<style>
th {{
    font-size: 20;
    white-space: nowrap;
}}
td {{
    font-size: 20;
}}
</style>
</head>
<body>
<h1><b>Вопрос</b>: {question}</h1>
<hr>
""")
n_result = 5
names = ['Default (all-MiniLM-L6-v2)', 'Embeddings', 'EmbeddingsGigaR', 'SbertLarge']

for i, item in enumerate(get_collection()): # доделать
    query_texts = item.coll.query(query_texts=[question], n_results=n_result)['documents'][0]
    f.write(f"""<table border>
    <th colspan="2">{item.name}</th>
""")
    for i, doc in enumerate(query_texts):
        f.write(f"""
                <tr>
                    <th>
                            <u>Ответ {i+1}:</u>
                    </th>
                    <td>
                        {doc.replace('\n', '<br>\n')}
                    </td>
                </tr>
            """)
    f.write("""
        </table>
        """)
f.write("""
        </body>
        </html>""")
