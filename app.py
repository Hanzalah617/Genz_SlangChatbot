import streamlit as st
import os
import csv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- Page Config ---
st.set_page_config(page_title="GenZ Slang Bot", page_icon="😎")
st.title("😎 GenZ Slang Chatbot")

# --- API Key ---
# Set 'GROQ_API_KEY' in Streamlit Cloud -> Settings -> Secrets
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("Please add GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

# --- Vector Store Logic (Updated for CSV) ---
@st.cache_resource
def load_vectorstore():
    file_path = "dataset.csv"
    
    if not os.path.exists(file_path):
        st.error(f"Error: '{file_path}' not found in the root directory. Please upload it to your GitHub repo.")
        st.stop()
    
    docs = []
    try:
        with open(file_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Combine all columns into a single string for the AI to read
                # Example: "slang: rizz, definition: charisma"
                content = " ".join([f"{k}: {v}" for k, v in row.items()])
                docs.append(Document(page_content=content))
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    if not docs:
        st.error("The CSV file is empty!")
        st.stop()

    # Using stable HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

# Initialize
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- RAG Chain (Using the working Llama 3.1 model) ---
llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=groq_api_key)

template = """
You are a GenZ slang expert. Use the following retrieved slang definitions to answer the question. 
Stay cool, helpful, and use a bit of GenZ vibe in your response.

Context: {context}
Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# The modern LCEL Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- UI ---
user_query = st.text_input("Drop a slang term here (e.g., 'What is rizz?'):")

if user_query:
    with st.spinner("Checking the files..."):
        try:
            response = rag_chain.invoke(user_query)
            st.markdown("### ✨ The Tea:")
            st.success(response)
        except Exception as e:
            st.error(f"Error: {e}")
