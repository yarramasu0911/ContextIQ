from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Query
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import json
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
    load_users,
    save_users,
    ROLES,
)
from .tokens import create_token, decode_token

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
    vector_store = PineconeVectorStore(dimension=embedding_model.dimension)
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

    # Rebuild BM25 from Pinecone so hybrid retrieval survives restarts.
    n_rebuilt = chunk_registry.rebuild_from_pinecone(vector_store)
    print(f"[startup] rebuilt BM25 index with {n_rebuilt} chunks from Pinecone")

    # Pre-warm the cross-encoder so the first /ask isn't a 20s cold start.
    if rag_chain.reranker is not None:
        try:
            rag_chain.reranker._ensure_loaded()
            print("[startup] reranker loaded")
        except Exception as e:
            print(f"[startup] reranker preload failed: {e}")

    print("Application loaded!")
    yield
    print("Shutting down application...")


app = FastAPI(title="ContextIQ RAG", lifespan=lifespan)


# ------------------------------------------------------------- viewer helper
def resolve_viewer(
    authorization: str | None = None,
    x_username: str | None = None,
    x_password: str | None = None,
) -> tuple[str, str]:
    """Authenticate via Bearer JWT (preferred) or X-Username/X-Password
    (fallback for clients that haven't migrated). Returns (username, role).
    """
    # Preferred: JWT
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        payload = decode_token(token)
        if payload is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        username = payload.get("sub")
        role = payload.get("role")
        if not username or not role:
            raise HTTPException(status_code=401, detail="Malformed token")
        # Re-check role server-side in case the user's role changed after
        # token issuance.
        current_role = get_role(username)
        if current_role is None:
            raise HTTPException(status_code=401, detail="Unknown user")
        return username, current_role

    # Fallback: basic headers
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
    role = get_role(username)
    token = create_token(username, role)
    return {"username": username, "role": role, "token": token}


@app.post("/auth/register")
def register(
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form("user"),
    authorization: str | None = Header(default=None),
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    if role not in ROLES:
        raise HTTPException(status_code=400, detail="Invalid role")
    if role != "user":
        _, acting_role = resolve_viewer(authorization, x_username, x_password)
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
    authorization: str | None = Header(default=None),
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    username, role = resolve_viewer(authorization, x_username, x_password)

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
    authorization: str | None = Header(default=None),
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    username, role = resolve_viewer(authorization, x_username, x_password)
    if vector_store.get_total_vectors(viewer_username=username, viewer_role=role) == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "No documents visible to you. Upload one first."},
        )
    return rag_chain.ask(question, viewer_username=username, viewer_role=role)


@app.post("/summarize")
def summarize_endpoint(
    authorization: str | None = Header(default=None),
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    username, role = resolve_viewer(authorization, x_username, x_password)
    if vector_store.get_total_vectors(viewer_username=username, viewer_role=role) == 0:
        return JSONResponse(
            status_code=400, content={"error": "No documents visible to you."}
        )
    return rag_chain.summarize(viewer_username=username, viewer_role=role)


@app.get("/documents")
def list_documents(
    authorization: str | None = Header(default=None),
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    username, role = resolve_viewer(authorization, x_username, x_password)
    return {
        "documents": vector_store.get_document_list(
            viewer_username=username, viewer_role=role
        )
    }


@app.post("/clear")
def clear_documents(
    scope: str = Query("mine"),
    authorization: str | None = Header(default=None),
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    username, role = resolve_viewer(authorization, x_username, x_password)
    if scope == "all" and role != "admin":
        raise HTTPException(
            status_code=403, detail="Only admins can clear all documents"
        )
    vector_store.clear(viewer_username=username, viewer_role=role, scope=scope)
    rag_chain.clear(viewer_username=username, viewer_role=role, scope=scope)
    return {"message": f"Cleared documents (scope={scope})."}


@app.get("/cache/stats")
def cache_stats(
    authorization: str | None = Header(default=None),
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    username, _ = resolve_viewer(authorization, x_username, x_password)
    if rag_chain.cache is None:
        return {"enabled": False}
    return {"enabled": True, **rag_chain.cache.stats(user_id=username)}


@app.post("/cache/clear")
def cache_clear(
    authorization: str | None = Header(default=None),
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    resolve_viewer(authorization, x_username, x_password)
    if rag_chain.cache is None:
        return {"enabled": False}
    rag_chain.cache.invalidate()
    return {"message": "Cache cleared."}


# -------------------------------------------------- streaming answer endpoint
@app.post("/ask/stream")
def ask_stream(
    question: str,
    authorization: str | None = Header(default=None),
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    username, role = resolve_viewer(authorization, x_username, x_password)
    if vector_store.get_total_vectors(viewer_username=username, viewer_role=role) == 0:
        raise HTTPException(status_code=400, detail="No documents visible to you.")

    # We stream the answer in two phases: (a) metadata event, (b) token chunks.
    # HF doesn't have native streaming here; we fall back to "pseudo-streaming"
    # by splitting the final answer into word chunks. OpenAI gets true token
    # streaming if LLM_PROVIDER=openai.
    result = rag_chain.ask(question, viewer_username=username, viewer_role=role)

    def event_stream():
        # SSE format: "data: {json}\n\n"
        meta = {
            k: v for k, v in result.items() if k not in ("answer", "answer_with_citations")
        }
        yield f"data: {json.dumps({'type': 'meta', 'payload': meta}, default=str)}\n\n"

        text = result.get("answer_with_citations") or result.get("answer") or ""
        for word in text.split():
            yield f"data: {json.dumps({'type': 'token', 'payload': word + ' '})}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ------------------------------------------------------- admin-only endpoints
def require_admin(viewer_role: str):
    if viewer_role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")


@app.get("/admin/users")
def admin_list_users(
    authorization: str | None = Header(default=None),
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    _, role = resolve_viewer(authorization, x_username, x_password)
    require_admin(role)
    users = load_users()
    return {
        "users": [
            {"username": u, "role": info.get("role", "user"), "active": info.get("active", True)}
            for u, info in users.items()
        ]
    }


@app.post("/admin/users/role")
def admin_change_role(
    username: str = Form(...),
    role: str = Form(...),
    authorization: str | None = Header(default=None),
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    _, acting_role = resolve_viewer(authorization, x_username, x_password)
    require_admin(acting_role)
    if role not in ROLES:
        raise HTTPException(status_code=400, detail="Invalid role")
    users = load_users()
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    users[username]["role"] = role
    save_users(users)
    return {"message": f"{username} is now {role}"}


@app.post("/admin/users/deactivate")
def admin_deactivate(
    username: str = Form(...),
    authorization: str | None = Header(default=None),
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    _, acting_role = resolve_viewer(authorization, x_username, x_password)
    require_admin(acting_role)
    users = load_users()
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    users[username]["active"] = False
    # Nuke the password so they can't log in even via old tokens once they expire.
    users[username]["password"] = "DEACTIVATED"
    save_users(users)
    return {"message": f"{username} deactivated"}


# --------------------------------------------------- observability endpoints
@app.get("/debug/trace")
def debug_trace(
    n: int = 20,
    trace_id: str | None = None,
    authorization: str | None = Header(default=None),
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    _, role = resolve_viewer(authorization, x_username, x_password)
    require_admin(role)
    if trace_id:
        entry = rag_chain.trace.find(trace_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="trace_id not found")
        return entry
    return {"traces": rag_chain.trace.recent(n=n)}


@app.get("/admin/audit")
def admin_audit(
    n: int = 100,
    authorization: str | None = Header(default=None),
    x_username: str | None = Header(default=None),
    x_password: str | None = Header(default=None),
):
    _, role = resolve_viewer(authorization, x_username, x_password)
    # HR and admin can read the audit log. Regular users cannot.
    if role not in ("admin", "hr"):
        raise HTTPException(status_code=403, detail="Admin/HR only")
    return {"entries": rag_chain.audit.tail(n=n)}
