import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Page Configuration ---
st.set_page_config(page_title="GenZ Slang Chatbot", page_icon="😎")
st.title("😎 GenZ Slang Chatbot")

# --- Load API Key ---
# Make sure to add GROQ_API_KEY in Streamlit Cloud Secrets
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("Please set the GROQ_API_KEY in your environment variables or Streamlit secrets.")
    st.stop()

# --- Vector Store Logic ---
@st.cache_resource
def load_vectorstore():
    # Ensure the path to your data is correct relative to your repo root
    try:
        with open("data/documents.txt", "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        st.error("Error: 'data/documents.txt' not found. Please check your file path.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.create_documents([text])

    # Using langchain-community/HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(docs, embeddings)

vectorstore = load_vectorstore()

# --- LLM and Chain Setup ---
llm = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=0.3, # A little bit of creativity for slang
    api_key=groq_api_key
)

# Define the Prompt (Essential for create_stuff_documents_chain)
prompt = ChatPromptTemplate.from_template("""
You are a GenZ slang expert. Use the context below to explain the slang terms.
If the answer isn't in the context, use your knowledge but keep the vibe cool.

Context:
{context}

Question: {input}

Answer:""")

retriever = vectorstore.as_retriever()
doc_chain = create_stuff_documents_chain(llm, prompt)
qa = create_retrieval_chain(retriever, doc_chain)

# --- User Interface ---
query = st.text_input("Ask a Gen Z slang question (e.g., 'What does skibidi mean?'):")

if query:
    with st.spinner("Vibing with the data..."):
        try:
            result = qa.invoke({"input": query})
            st.markdown("### 😎 The Tea:")
            st.success(result["answer"])
        except Exception as e:
            st.error(f"An error occurred: {e}")
