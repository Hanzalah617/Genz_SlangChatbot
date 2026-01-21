import streamlit as st
import os

# Essential LangChain Components
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# These are the modern, direct import paths
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# --- Page Config ---
st.set_page_config(page_title="GenZ Slang Bot", page_icon="😎")
st.title("😎 GenZ Slang Chatbot")

# --- API Key Setup ---
# Set 'GROQ_API_KEY' in Streamlit Cloud -> Settings -> Secrets
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("Missing GROQ_API_KEY. Please add it to Streamlit Secrets.")
    st.stop()

# --- Vector Store Logic ---
@st.cache_resource
def load_vectorstore():
    # Path handling for Streamlit Cloud
    data_path = "data/documents.txt"
    
    if not os.path.exists(data_path):
        st.error(f"File not found: {data_path}. Please check your GitHub folder structure.")
        st.stop()
    
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])

    # Using langchain-huggingface for better stability
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    return FAISS.from_documents(docs, embeddings)

# Initialize
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

# --- Chain Setup ---
llm = ChatGroq(model_name="llama3-8b-8192", api_key=groq_api_key)

# The 'context' and 'input' variables are required by the retrieval chains
prompt = ChatPromptTemplate.from_template("""
You are a GenZ slang expert. Use the context below to explain the terms.
Context:
{context}

Question: {input}

Answer:""")

# Create the chains using the new standard
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# --- UI ---
user_query = st.text_input("What slang do you need translated?")

if user_query:
    with st.spinner("Searching the vibes..."):
        try:
            response = rag_chain.invoke({"input": user_query})
            st.markdown("### ✨ The Tea:")
            st.success(response["answer"])
        except Exception as e:
            st.error(f"Something went wrong: {e}")
