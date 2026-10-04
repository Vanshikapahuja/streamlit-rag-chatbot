import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

# load environment variables from .env file
load_dotenv()

working_dir = os.path.dirname(os.path.abspath((__file__)))

# Load the Embedding model
embedding = HuggingFaceEmbeddings()

# Load the Groq model
llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature = 0
)

def process_documents_to_chromadb(filename):
    # Load the PDF document using UnstructuredPDFLoader
    loader = UnstructuredPDFLoader(f"{working_dir}/{filename}")
    documents = loader.load()

    # Split text into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    # store the document chunks into Chroma vector database
    vector_db = Chroma.from_documents(
        documents = texts,
        embedding = embedding,
        persist_directory = f"{working_dir}/doc_vectorstore"
    )

    return 0

def answer_question(user_question):
    # load the persistent chroma vector database
    vector_db = Chroma(
        persist_directory = f"{working_dir}/doc_vectorstore",
        embedding_function = embedding
    )

    # create a retriever for document search
    retriever = vector_db.as_retriever()

    # Create a RetrievalQA chain to answer user questions using LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = retriever
    )

    response = qa_chain.invoke({"query":user_question})
    answer = response["result"]

    return answer



