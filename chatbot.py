import os
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

def load_documents():
    docs = []

    # Load cleaned scholarship data
    if os.path.exists("data/cleaned_data.csv"):
        df = pd.read_csv("data/cleaned_data.csv")
        for _, row in df.iterrows():
            text = f"{row.get('title', '')}\n{row.get('description', '')}"
            docs.append(Document(page_content=text.strip()))

    # Load previous Q&A history
    if os.path.exists("data/qna_history.csv"):
        qna = pd.read_csv("data/qna_history.csv")
        for _, row in qna.iterrows():
            q = row.get("question", "")
            a = row.get("answer", "")
            docs.append(Document(page_content=f"Q: {q}\nA: {a}"))

    return docs

def get_chain():
    documents = load_documents()

    # Optional chunking
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(documents)

    # Embed and store
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

    return qa_chain

qa_chain = get_chain()

def ask_bot(query):
    return qa_chain.run(query)
