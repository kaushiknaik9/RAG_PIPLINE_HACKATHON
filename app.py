import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

st.title("Local RAG over PDF (Ollama + Qdrant)")

# Config from .env
QDRANT_URL = os.getenv("QDRANT_URL", ":memory:")  # Default to in-memory for dev
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")

# 1) Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Save uploaded file to a temp path so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    st.success(f"Loaded file: {uploaded_file.name}")

    # 2) Load and chunk
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    # 3) Build vector store (in-memory or remote via .env)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_db = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,  # ":memory:", "http://localhost:6333", or cloud URL
        collection_name=COLLECTION_NAME,
        prefer_grpc=False,
    )

    llm = ChatOllama(model=LLM_MODEL)

    # 4) Ask question
    user_query = st.text_input("Ask something about this PDF:")

    if st.button("Ask") and user_query.strip():
        search_results = vector_db.similarity_search(user_query, k=4)

        content = "\n\n".join(
            [
                f"Page Content: {r.page_content}\n"
                f"Page Number: {r.metadata.get('page_label')}\n"
                f"File Location: {r.metadata.get('source')}"
                for r in search_results
            ]
        )

        system_prompt = f"""
        You are a helpful AI Assistant who answers user queries based on the available
        content retrieved from a PDF file along with page contents and page numbers.
        You should only answer the user based on the following content and navigate
        the user to open the right page number to know more.

        Content:
        {content}
        """

        with st.spinner("Thinking..."):
            resp = llm.invoke(
                [
                    ("system", system_prompt),
                    ("user", user_query),
                ]
            )

        st.subheader("Answer")
        st.write(resp.content)
