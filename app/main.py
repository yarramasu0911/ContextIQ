from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import shutil
import os

from .document_processor import process_document
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .rag_chain import RAGChain


UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models when server starts."""
    global embedding_model, vector_store, rag_chain
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(dimension=384)
    rag_chain = RAGChain(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm_provider="huggingface",
    )
    print("Application loaded!")
    yield
    print("Shutting down application...")


app = FastAPI(title="RAG Document Q&A", lifespan=lifespan)


@app.get("/health/")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    # 1. Validate file type
    if not file.filename.endswith(
        (".pdf", ".txt", ".docx", ".csv", ".html", ".htm", ".json", ".rtf", ".md")
    ):
        return JSONResponse(
            status_code=400,
            content={"error": "Unsupported file type. Upload PDF or TXT."},
        )

    # 2. Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 3. Process into chunks
    chunks = process_document(file_path)

    # 4. Create embeddings and store
    embeddings = embedding_model.embed_texts(chunks)
    vector_store.add_documents(chunks, embeddings)

    return {
        "message": f"Document '{file.filename}' processed successfully",
        "chunks": len(chunks),
        "total_documents_stored": vector_store.index.ntotal,
    }


@app.post("/ask")
def ask_question(question: str):
    """Ask a question about uploaded documents."""
    if vector_store.index.ntotal == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "No documents uploaded yet. Upload a document first."},
        )

    result = rag_chain.ask(question)
    return result
