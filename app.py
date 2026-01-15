import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.title("🔍 PDF RAG - Works Everywhere")

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    st.success(f"✅ {uploaded_file.name} loaded")

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatOllama(model=LLM_MODEL)

    query = st.text_input("❓ Ask about PDF:")
    if st.button("💡 Answer") and query:
        docs_found = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs_found])

        prompt = f"Answer using ONLY this PDF:\n\n{context}\n\nQ: {query}"

        with st.spinner("🧠 Answering..."):
            response = llm.invoke(prompt)

        st.subheader("📖 Answer")
        st.write(response.content)

st.caption("🚀 Powered by FAISS + Ollama | Works on mobile")
