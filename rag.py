from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore

# 1) Embeddings – must match indexing model
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

# 2) Connect to existing Qdrant collection "rag"
vector_db = QdrantVectorStore(
    collection_name="rag",
    url="http://localhost:6333",
    embedding=embedding_model,
    prefer_grpc=False,
)

# 3) Local LLM for generation
llm = ChatOllama(model="llama3.2:3b")

# 4) Ask user, retrieve chunks, and answer
user_query = input("Ask something: ")

search_results = vector_db.similarity_search(user_query, k=4)

content = "\n\n".join(
    [
        f"Page Content: {result.page_content}\n"
        f"Page Number: {result.metadata.get('page_label')}\n"
        f"File Location: {result.metadata.get('source')}"
        for result in search_results
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

resp = llm.invoke(
    [
        ("system", system_prompt),
        ("user", user_query),
    ]
)

print("\n\nAnswer:\n")
print(resp.content)
