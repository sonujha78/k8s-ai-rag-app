from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import time

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

load_dotenv()
app = FastAPI()

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.minikube.internal:11434")
COLLECTION_NAME = "learning_vectors"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

QUERY_COUNT = Counter("rag_queries_total", "Total number of queries received")
QUERY_LATENCY = Histogram("rag_query_latency_seconds", "Time taken to answer a query")


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "rag-query-service",
        "qdrant_url": QDRANT_URL,
        "ollama_url": OLLAMA_BASE_URL,
        "llm_model": LLM_MODEL
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/query")
async def query(request: QueryRequest):
    QUERY_COUNT.inc()
    start_time = time.time()

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL
    )

    results = vector_store.similarity_search(request.question, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    prompt = f"""Answer the question based only on the following context:

{context}

Question: {request.question}
"""

    response = llm.invoke(prompt)

    QUERY_LATENCY.observe(time.time() - start_time)

    return {
        "answer": response.content,
        "sources_used": len(results)
    }
