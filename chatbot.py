import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load your CSV data
loader = CSVLoader(file_path="data/cleaned_data.csv", encoding="utf-8")
docs = loader.load()

# Split into smaller chunks (so embeddings are better)
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs_split = splitter.split_documents(docs)

# Create vector store using OpenAI embeddings
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embedding)

# Load OpenAI LLM (GPT-3.5)
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # or "gpt-4"
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create retriever and chain
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Function to query the bot
def ask_bot(query):
    return qa_chain.run(query)
