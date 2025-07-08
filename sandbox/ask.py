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
query_texts = []
names = ['Default (all-MiniLM-L6-v2)', 'Embeddings', 'EmbeddingsGigaR', 'SbertLarge']

for name, coll in zip(names, get_collection()): # доделать
    query_texts.append(coll.query(query_texts=[question], n_results=n_result)['documents'][0])
for i in range(len(names)):
    if i == 0:
        f.write(f"""<table border>
    <th colspan="2">default</th>
""")
    else:
        f.write(f"""<table border>
    <th colspan="2">{names[i]}</th>
""")
    for n in range(5):
        f.write(f"""
                <tr>
                    <th>
                            <u>Ответ {n+1}:</u>
                    </th>
                    <td>
                        {query_texts[i][n].replace('\n', '<br>\n')}
                    </td>
                </tr>
            """)
    f.write("""
        </table>
        </body>
        </html>
        """)