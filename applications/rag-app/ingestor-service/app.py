from fastapi import FastAPI, UploadFile
from dotenv import load_dotenv
import shutil
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()
app = FastAPI()

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.minikube.internal:11434")
COLLECTION_NAME = "learning_vectors"
EMBEDDING_MODEL = "nomic-embed-text"


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "rag-ingestor-service",
        "qdrant_url": QDRANT_URL,
        "ollama_url": OLLAMA_BASE_URL,
        "embedding_model": EMBEDDING_MODEL
    }


@app.post("/ingest")
async def ingest_pdf(file: UploadFile):
    file_path = f"/tmp/{file.filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400
    )
    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME
    )

    return {"status": "indexed", "chunks": len(chunks)}
