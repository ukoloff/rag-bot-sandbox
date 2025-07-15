from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from langchain_core.runnables.config import RunnableConfig
from dotenv import load_dotenv
from os.path import normpath, join, dirname
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter
from langchain_chroma import Chroma


load_dotenv()

llm = GigaChat()

path_to_db = normpath(join(dirname(__file__), '..', '..', 'chroma.kb'))
db = Chroma(collection_name="kb.gigaRtext", embedding_function=GigaChatEmbeddings(model='Embeddings'), persist_directory=path_to_db)
retriever = db.as_retriever()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Question-Answering chatbot. Please provide answers to the given questions.",
        ),
        # Use "chat_history" as the key for conversation history without modifying it if possible.
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "#Context:\n{context}\n#Question:\n{question}"),  # Use user input as a variable.
    ]
)

def join(docs):
    s = '\n\n'.join(doc.page_content for doc in docs)
    return s

def view(x):
    print(">>>", x)
    return x

chain = (
    {
        "context": itemgetter("question") | retriever | join,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | prompt
    | view
    | llm
    | StrOutputParser()
)

store = {}


# A function to retrieve session history based on the session ID.
def get_session_history(session_ids):
    print(f"[Conversation session ID]: {session_ids}")
    if session_ids not in store:  # When the session ID is not in the store.
        # Create a new ChatMessageHistory object and save it in the store.
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # Return the session history for the given session ID.

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # A function to retrieve session history.
    input_messages_key="question",  # The key where the user's question will be inserted into the template variable.
    history_messages_key="chat_history",  # The key for the message in the history.
)

r1 = chain_with_history.invoke(
    # Question input.
    {"question": "My name is Teddy.", "context": "AAA"},
    # Record conversations based on the session ID.
    config=RunnableConfig(configurable={"session_id": "abc123"})
)
print(r1)

r2 = chain_with_history.invoke(
    # Question input.
    {"question": "What's my name?", "context": "BBB"},
    # Record conversations based on the session ID.
    config=RunnableConfig(configurable={"session_id": "abc123"})
)
print(r2)