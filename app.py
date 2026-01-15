import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.title("PDF Search")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    st.success(f"✅ {uploaded_file.name}")

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    query = st.text_input("Search PDF:")
    if st.button("Find") and query:
        results = []
        for i, chunk in enumerate(chunks):
            if query.lower() in chunk.page_content.lower():
                results.append(
                    {
                        "chunk": chunk.page_content[:300] + "...",
                        "page": chunk.metadata.get("page", i),
                        "score": chunk.page_content.lower().count(query.lower()),
                    }
                )

        if results:
            st.subheader("Matches:")
            for r in sorted(results, key=lambda x: x["score"], reverse=True)[:5]:
                st.write(f"**Page {r['page']}** (matches: {r['score']})")
                st.write(r["chunk"])
        else:
            st.write("No matches found")

st.caption("Pure Python - Works everywhere")
