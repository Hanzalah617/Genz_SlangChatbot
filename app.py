import streamlit as st
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

st.set_page_config(page_title="GenZ Slang Chatbot")

st.title("😎 GenZ Slang Chatbot")

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

    return FAISS.from_documents(docs, embeddings)

vectorstore = load_vectorstore()

llm = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

query = st.text_input("Ask a Gen Z slang question:")

if query:
    with st.spinner("Thinking..."):
        st.success(qa.run(query))
