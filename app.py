import streamlit as st
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Page config
st.set_page_config(page_title="Groq RAG Chatbot", layout="centered")

st.title("⚡ Groq RAG-based Chatbot")

# Load & embed data
@st.cache_resource
def load_vectorstore():
    with open("data/documents.txt", "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

vectorstore = load_vectorstore()

# Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-8b-8192",
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# Chat UI
query = st.text_input("Ask something from your data:")

if query:
    with st.spinner("Thinking..."):
        answer = qa_chain.run(query)
        st.success(answer)
