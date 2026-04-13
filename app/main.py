from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import shutil
import os

from .document_processor import process_document
from .embeddings import EmbeddingModel
from .vector_store_pinecone import PineconeVectorStore
from .rag_chain import RAGChain

embedding_model = None
vector_store = None
rag_chain = None

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, vector_store, rag_chain
    embedding_model = EmbeddingModel()
    vector_store = PineconeVectorStore()
    rag_chain = RAGChain(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm_provider="huggingface",
    )
    print("Application loaded!")
    yield
    print("Shutting down application...")


app = FastAPI(title="RAG Document Q&A", lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(
        (".pdf", ".txt", ".docx", ".csv", ".html", ".htm", ".json", ".rtf", ".md")
    ):
        return JSONResponse(
            status_code=400,
            content={"error": "Unsupported file type."},
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = process_document(file_path)
    chunk_texts = [chunk["content"] for chunk in chunks]
    embeddings = embedding_model.embed_texts(chunk_texts)
    vector_store.add_documents(chunks, embeddings, filename=file.filename)

    return {
        "message": f"Document '{file.filename}' processed successfully",
        "chunks": len(chunks),
        "total_documents_stored": vector_store.get_total_vectors(),
    }


@app.post("/ask")
def ask_question(question: str):
    if vector_store.get_total_vectors() == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "No documents uploaded yet."},
        )
    return rag_chain.ask(question)