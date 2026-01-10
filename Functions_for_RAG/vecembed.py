from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from Functions_for_RAG.chunk import splitted


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vec = embeddings.embed_query("Llamas are members of the camelid family")
print(len(vec), vec[:5])

vector_store = QdrantVectorStore.from_documents(
    documents=splitted,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="rag",
)
print("Indexing done")
