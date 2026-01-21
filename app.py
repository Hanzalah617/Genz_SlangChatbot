import streamlit as st
import os

# Core LangChain imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# These are the specific imports that were failing
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# --- Page Config ---
st.set_page_config(page_title="GenZ Slang Bot", page_icon="😎")
st.title("😎 GenZ Slang Chatbot")

# --- API Key Setup ---
# Set this in Streamlit Cloud -> Settings -> Secrets
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("Missing GROQ_API_KEY. Please add it to Streamlit Secrets.")
    st.stop()

# --- Load and Process Data ---
@st.cache_resource
def load_vectorstore():
    data_path = "data/documents.txt"
    
    # Check if directory exists
    if not os.path.exists("data"):
        os.makedirs("data")
        
    # Check if file exists, if not create a sample
    if not os.path.exists(data_path):
        with open(data_path, "w", encoding="utf-8") as f:
            f.write("Skibidi: A nonsense word. Rizz: Charisma. Gyatt: An exclamation.")
    
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])

    # Using the updated HuggingFaceEmbeddings class
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    return FAISS.from_documents(docs, embeddings)

# Initialize Vectorstore
try:
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()
except Exception as e:
    st.error(f"Failed to load vectorstore: {e}")
    st.stop()

# --- RAG Chain Setup ---
llm = ChatGroq(model_name="llama3-8b-8192", api_key=groq_api_key)

# The prompt is required for the chains to link correctly
prompt = ChatPromptTemplate.from_template("""
Answer the user's question based ONLY on the context below. 
If you don't know, say you're 'not vibing with that info yet'.

Context:
{context}

Question: {input}
""")

# Build the chains
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# --- UI ---
user_input = st.text_input("What slang are we looking up?")

if user_input:
    with st.spinner("Searching the archives..."):
        try:
            response = rag_chain.invoke({"input": user_input})
            st.markdown("### ✨ The Tea:")
            st.success(response["answer"])
        except Exception as e:
            st.error(f"Chain error: {e}")
