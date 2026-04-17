from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Query
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import shutil
import os

from .document_processor import process_document
from .embeddings import EmbeddingModel
from .vector_store_pinecone import PineconeVectorStore
from .rag_chain import RAGChain
from .retrieval import ChunkRegistry
from .auth import (
    authenticate_user,
    register_user,
    get_role,
    default_visibility_for,
    visibilities_allowed_for_upload,
    ROLES,
)

embedding_model = None
vector_store = None
rag_chain = None
chunk_registry = None

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, vector_store, rag_chain, chunk_registry
    embedding_model = EmbeddingModel()
    vector_store = PineconeVectorStore()
    chunk_registry = ChunkRegistry()
    rag_chain = RAGChain(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm_provider=os.getenv("LLM_PROVIDER", "huggingface"),
        registry=chunk_registry,
        enable_rerank=os.getenv("ENABLE_RERANK", "1") == "1",
        enable_cache=os.getenv("ENABLE_CACHE", "1") == "1",
        enable_hyde=os.getenv("ENABLE_HYDE", "0") == "1",
        enable_multi_query=os.getenv("ENABLE_MULTI_QUERY", "0") == "1",
        enable_faithfulness=os.getenv("ENABLE_FAITHFULNESS", "1") == "1",
    )
    print("Application loaded!")
    yield
    print("Shutting down application...")


app = FastAPI(title="ContextIQ RAG", lifespan=lifespan)


# ------------------------------------------------------------- viewer helper
def resolve_viewer(
    x_username: str | None, x_password: str | None
) -> tuple[str, str]:
    """Authenticate via simple headers, return (username, role).

    Uses request headers `X-Username` / `X-Password`. Role is looked up from
    auth.py — frontend can't spoof it.
    """
    if not x_username or not x_password:
        raise HTTPException(status_code=401, detail="Missing credentials")
    if not authenticate_user(x_username, x_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    role = get_role(x_username)
    if role is None:
        raise HTTPException(status_code=401, detail="User has no role")
    return x_username, role


# --------------------------------------------------------------------- routes
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/auth/login")
def login(username: str = Form(...), password: str = Form(...)):
    if not authenticate_user(username, password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"username": username, "role": get_role(username)}


@app.post("/auth/register")
def register(
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form("user"),
    # Only admins may create admin/hr accounts; pass creds in headers.
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    if role not in ROLES:
        raise HTTPException(status_code=400, detail="Invalid role")
    if role != "user":
        _, acting_role = resolve_viewer(x_username, x_password)
        if acting_role != "admin":
            raise HTTPException(
                status_code=403,
                detail="Only admins can create admin/hr accounts",
            )
    ok = register_user(username, password, role=role)
    if not ok:
        raise HTTPException(
            status_code=400, detail="Username already exists or invalid role"
        )
    return {"message": f"Registered {username} as {role}"}


SUPPORTED_EXTS = (
    ".pdf", ".txt", ".docx", ".doc", ".csv", ".tsv",
    ".html", ".htm", ".json", ".rtf", ".md",
    ".xlsx", ".xls", ".pptx", ".ppt",
    ".log", ".yaml", ".yml", ".xml",
)


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    visibility: str | None = Form(default=None),
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    username, role = resolve_viewer(x_username, x_password)

    if not file.filename.lower().endswith(SUPPORTED_EXTS):
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTS)}"
            },
        )

    # visibility: fall back to role default, then validate
    visibility = visibility or default_visibility_for(role)
    allowed = visibilities_allowed_for_upload(role)
    if visibility not in allowed:
        raise HTTPException(
            status_code=403,
            detail=f"Role '{role}' may only upload with visibility in {allowed}",
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = process_document(file_path)
    chunk_texts = [chunk["content"] for chunk in chunks]
    embeddings = embedding_model.embed_texts(chunk_texts)
    vector_store.add_documents(
        chunks,
        embeddings,
        filename=file.filename,
        owner_username=username,
        owner_role=role,
        visibility=visibility,
    )
    rag_chain.register_chunks(
        chunks,
        filename=file.filename,
        owner_username=username,
        owner_role=role,
        visibility=visibility,
    )

    return {
        "message": f"Document '{file.filename}' processed successfully",
        "chunks": len(chunks),
        "visibility": visibility,
        "total_visible_chunks": vector_store.get_total_vectors(
            viewer_username=username, viewer_role=role
        ),
    }


@app.post("/ask")
def ask_question(
    question: str,
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    username, role = resolve_viewer(x_username, x_password)
    if vector_store.get_total_vectors(viewer_username=username, viewer_role=role) == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "No documents visible to you. Upload one first."},
        )
    return rag_chain.ask(question, viewer_username=username, viewer_role=role)


@app.post("/summarize")
def summarize_endpoint(
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    username, role = resolve_viewer(x_username, x_password)
    if vector_store.get_total_vectors(viewer_username=username, viewer_role=role) == 0:
        return JSONResponse(
            status_code=400, content={"error": "No documents visible to you."}
        )
    return rag_chain.summarize(viewer_username=username, viewer_role=role)


@app.get("/documents")
def list_documents(
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    username, role = resolve_viewer(x_username, x_password)
    return {
        "documents": vector_store.get_document_list(
            viewer_username=username, viewer_role=role
        )
    }


@app.post("/clear")
def clear_documents(
    scope: str = Query("mine"),
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    username, role = resolve_viewer(x_username, x_password)
    if scope == "all" and role != "admin":
        raise HTTPException(
            status_code=403, detail="Only admins can clear all documents"
        )
    vector_store.clear(viewer_username=username, viewer_role=role, scope=scope)
    rag_chain.clear(viewer_username=username, viewer_role=role, scope=scope)
    return {"message": f"Cleared documents (scope={scope})."}


@app.get("/cache/stats")
def cache_stats(
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    username, _ = resolve_viewer(x_username, x_password)
    if rag_chain.cache is None:
        return {"enabled": False}
    return {"enabled": True, **rag_chain.cache.stats(user_id=username)}


@app.post("/cache/clear")
def cache_clear(
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    resolve_viewer(x_username, x_password)
    if rag_chain.cache is None:
        return {"enabled": False}
    rag_chain.cache.invalidate()
    return {"message": "Cache cleared."}
