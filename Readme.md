# RAG Pipeline with Ollama, Qdrant and Streamlit

This project is a local RAG (Retrieval‑Augmented Generation) pipeline that lets you upload a PDF and ask questions about it using a local LLM (`llama3.2:3b` via Ollama) plus a vector database (Qdrant).[file:1][web:12]

---

## Features

- Local LLM inference with **Ollama** (`llama3.2:3b`).[web:12]
- Text embeddings with **mxbai-embed-large** via `langchain-ollama`.[web:32]
- Vector search over PDF chunks using **Qdrant**.[file:1]
- Simple **Streamlit** web UI to upload a PDF and chat with it.[web:57]

---

## Project structure

```text
RAG_PIPELINE_HACKATHON/
│
├── Functions_for_RAG/
│   ├── __init__.py
│   ├── loaddata.py        # CLI PDF loader (sample)
│   ├── chunk.py           # CLI chunking (sample)
│   ├── vecembed.py        # CLI indexing into Qdrant
│   └── RAG_SAMPLE_DATA.pdf
│
├── RAG/
│   ├── app.py             # Streamlit web app (RAG over PDF)
│   └── docker_compose.yml # Qdrant service
│
├── rag.py                 # CLI RAG (ask from terminal)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Prerequisites

1. **Python** 3.10+ and virtualenv.  
2. **Ollama** installed and running on your machine.[web:20]  
   - Required models:
     ```bash
     ollama pull llama3.2:3b
     ollama pull mxbai-embed-large
     ```
3. **Docker** installed and running (for Qdrant).[web:69]

---

## Setup

```bash
# In project root
python -m venv RAG
RAG\Scripts\activate          # On Windows

pip install -r requirements.txt
```

Start Qdrant:

```bash
cd RAG
docker compose up -d
cd ..
```

Make sure the Ollama service is running and the two models are pulled.

---

## Indexing (build the vector store once)

For the sample PDF `RAG_SAMPLE_DATA.pdf` inside `Functions_for_RAG`:

```bash
RAG\Scripts\activate
python -m Functions_for_RAG.vecembed
```

This script:

- Loads the PDF with `PyPDFLoader`.  
- Splits it into chunks with `RecursiveCharacterTextSplitter`.  
- Creates embeddings using `OllamaEmbeddings(model="mxbai-embed-large")`.  
- Stores them in a Qdrant collection named `"rag"` on `http://localhost:6333`.[file:1]

Run this again only when documents change.

---

## CLI RAG (terminal usage)

To query the existing `"rag"` collection from the terminal:

```bash
python rag.py
```

Flow:

1. Connects to Qdrant `"rag"` using `OllamaEmbeddings("mxbai-embed-large")`.  
2. Prompts for `user_query = input("Ask something: ")`.  
3. Retrieves top‑k chunks via `similarity_search`.  
4. Builds a system prompt including page content, page numbers and file paths.  
5. Calls `ChatOllama(model="llama3.2:3b")` and prints the answer.

---

## Web app (Streamlit)

### Run the app

```bash
streamlit run app.py
```

Open the URL logged by Streamlit (usually `http://localhost:8501`).[web:61]

### How the app works

In `app.py`:

1. `st.file_uploader("Upload a PDF", type="pdf")` lets the user upload a PDF.[web:74]
2. The file is written to a temporary path and loaded with `PyPDFLoader`.  
3. Text is split into chunks with `RecursiveCharacterTextSplitter`.  
4. Chunks are embedded with `OllamaEmbeddings("mxbai-embed-large")`.  
5. A Qdrant collection (e.g. `"rag"`) is built or updated with these chunks.  
6. The user types a question; the app retrieves relevant chunks via `similarity_search`.  
7. A system prompt is built from those chunks and sent to `ChatOllama("llama3.2:3b")`.  
8. The answer is displayed in the browser.

This gives a fully local “Chat with your PDF” workflow using Ollama + Qdrant + Streamlit.

---

## Possible extensions

- Support multiple PDFs by creating per‑file collections (`rag_<filename>`) or using document‑level metadata.  
- Add conversation history and user identity (MongoDB / LangGraph style) for longer chats.[file:1]  
- Swap in other Ollama models (DeepSeek, Qwen, Gemma, etc.) by changing the `ChatOllama` and embedding model names.[web:32][web:40]
```
