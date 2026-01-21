import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# Update these specific lines:
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- Page Configuration ---
st.set_page_config(page_title="GenZ Slang Chatbot", page_icon="😎")
st.title("😎 GenZ Slang Chatbot")

# --- Load API Key ---
# Add your GROQ_API_KEY in the Streamlit Cloud Secrets setting
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.warning("Please add your GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

# --- Vector Store Logic ---
@st.cache_resource
def load_vectorstore():
    # Make sure your data folder is in your GitHub repo
    if not os.path.exists("data/documents.txt"):
        st.error("Missing 'data/documents.txt' file!")
        st.stop()

    with open("data/documents.txt", "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.create_documents([text])

    # This handles the embedding model download automatically
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(docs, embeddings)

vectorstore = load_vectorstore()

# --- LLM and Chain Setup ---
llm = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=0.4,
    api_key=groq_api_key
)

# A prompt template is REQUIRED for create_stuff_documents_chain to work
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
If the answer isn't in the context, use your inner GenZ knowledge but stay helpful.

Context:
{context}

Question: {input}

Answer:""")

retriever = vectorstore.as_retriever()

# Build the RAG chain
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# --- Chat Interface ---
query = st.text_input("Ask about some brainrot or slang:")

if query:
    with st.spinner("Cooking..."):
        try:
            response = retrieval_chain.invoke({"input": query})
            st.markdown("### ✨ Response:")
            st.write(response["answer"])
        except Exception as e:
            st.error(f"Something went wrong: {e}")
