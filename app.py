import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Page Config ---
st.set_page_config(page_title="GenZ Slang Bot", page_icon="😎")
st.title("😎 GenZ Slang Chatbot")

# --- API Key ---
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Please add GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

# --- Vector Store ---
@st.cache_resource
def load_vectorstore():
    data_path = "data/documents.txt"
    if not os.path.exists(data_path):
        os.makedirs("data", exist_ok=True)
        with open(data_path, "w") as f:
            f.write("Skibidi: Bad or cool. Rizz: Charisma. Gyatt: Wow. Sigma: Alpha/Leader.")

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

# --- The RAG Chain ---
# UPDATED MODEL NAME HERE: llama-3.1-8b-instant
llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=groq_api_key)

template = """
You are a GenZ slang expert. Use the context to answer the question. 
If you don't know, say you're 'not vibing with that info'.

Context: {context}
Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- UI ---
user_query = st.text_input("What slang do you need translated?")

if user_query:
    with st.spinner("Cooking with Llama 3.1..."):
        try:
            response = rag_chain.invoke(user_query)
            st.markdown("### ✨ The Tea:")
            st.success(response)
        except Exception as e:
            st.error(f"Error: {e}")
