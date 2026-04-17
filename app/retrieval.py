"""Hybrid retrieval: BM25 (keyword) + dense (Pinecone) fused via Reciprocal Rank
Fusion, then optionally reranked with a cross-encoder.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from .access import can_see


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _chunk_key(filename: str, text: str) -> str:
    h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{filename}::{h}"


class ChunkRegistry:
    """In-memory BM25 index over all uploaded chunks, with role/visibility
    awareness. Chunks live in a single global bucket; viewers are filtered at
    query time via `access.can_see` so the same chunk can surface for anyone
    allowed to see it without being duplicated across buckets.
    """

    def __init__(self):
        self._chunks: list[dict] = []  # {text, filename, owner_username, owner_role, visibility, key}
        self._bm25: BM25Okapi | None = None

    def add(
        self,
        texts: list[str],
        filename: str,
        owner_username: str = "default",
        owner_role: str = "user",
        visibility: str = "private",
    ):
        for text in texts:
            self._chunks.append(
                {
                    "text": text,
                    "filename": filename,
                    "owner_username": owner_username,
                    "owner_role": owner_role,
                    "visibility": visibility,
                    "key": _chunk_key(filename, text),
                }
            )
        self._rebuild_bm25()

    def clear(
        self,
        viewer_username: str = "default",
        viewer_role: str = "admin",
        scope: str = "mine",
    ):
        if scope == "all" and viewer_role == "admin":
            self._chunks = []
        else:
            self._chunks = [
                c for c in self._chunks if c["owner_username"] != viewer_username
            ]
        self._rebuild_bm25()

    def _rebuild_bm25(self):
        if not self._chunks:
            self._bm25 = None
            return
        tokenized = [_tokenize(c["text"]) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

    def bm25_search(
        self,
        query: str,
        top_k: int,
        viewer_username: str = "default",
        viewer_role: str = "admin",
    ) -> list[tuple[str, float, str]]:
        if self._bm25 is None or not self._chunks:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        if len(scores) == 0:
            return []

        order = np.argsort(scores)[::-1]
        results: list[tuple[str, float, str]] = []
        for idx in order:
            if scores[idx] <= 0:
                continue
            c = self._chunks[idx]
            if not can_see(
                viewer_username,
                viewer_role,
                c["owner_username"],
                c["visibility"],
            ):
                continue
            results.append((c["text"], float(scores[idx]), c["filename"]))
            if len(results) >= top_k:
                break
        return results

    def size(
        self, viewer_username: str = "default", viewer_role: str = "admin"
    ) -> int:
        return sum(
            1
            for c in self._chunks
            if can_see(
                viewer_username, viewer_role, c["owner_username"], c["visibility"]
            )
        )


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float, str]]],
    k: int = 60,
) -> list[tuple[str, float, str]]:
    """Merge multiple ranked lists of (text, score, filename) tuples via RRF.

    RRF ignores raw scores and uses only ranks, which makes BM25 and cosine
    scores safely combinable despite living on totally different scales.
    """
    fused: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, (text, _score, filename) in enumerate(ranked):
            key = _chunk_key(filename, text)
            entry = fused.setdefault(
                key, {"text": text, "filename": filename, "score": 0.0}
            )
            entry["score"] += 1.0 / (k + rank + 1)

    merged = sorted(fused.values(), key=lambda e: e["score"], reverse=True)
    return [(e["text"], e["score"], e["filename"]) for e in merged]


class Reranker:
    """Cross-encoder reranker. Lazy-loads the model to keep startup cheap."""

    _MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self._MODEL_NAME
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float, str]],
        top_k: int,
    ) -> list[tuple[str, float, str]]:
        if not candidates:
            return []
        self._ensure_loaded()
        pairs = [(query, text) for text, _score, _fn in candidates]
        scores = self._model.predict(pairs)
        order = np.argsort(scores)[::-1][:top_k]
        return [
            (candidates[i][0], float(scores[i]), candidates[i][2]) for i in order
        ]


class HybridRetriever:
    """Orchestrates dense (Pinecone) + sparse (BM25) retrieval with RRF fusion
    and optional cross-encoder reranking.
    """

    def __init__(
        self,
        vector_store,
        embedding_model,
        registry: ChunkRegistry,
        reranker: Optional[Reranker] = None,
        candidate_pool: int = 20,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.registry = registry
        self.reranker = reranker
        self.candidate_pool = candidate_pool

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        viewer_username: str = "default",
        viewer_role: str = "admin",
        extra_queries: Optional[list[str]] = None,
    ) -> list[tuple[str, float, str]]:
        """Return top_k (text, score, filename) tuples.

        extra_queries lets callers add HyDE / multi-query variants — each one
        contributes its own ranked list to RRF.
        """
        queries = [query] + (extra_queries or [])

        dense_lists: list[list[tuple[str, float, str]]] = []
        for q in queries:
            emb = self.embedding_model.embed_query(q)
            dense_lists.append(
                self.vector_store.search(
                    emb,
                    top_k=self.candidate_pool,
                    viewer_username=viewer_username,
                    viewer_role=viewer_role,
                )
            )

        sparse = self.registry.bm25_search(
            query,
            top_k=self.candidate_pool,
            viewer_username=viewer_username,
            viewer_role=viewer_role,
        )

        fused = reciprocal_rank_fusion(dense_lists + [sparse])

        if not fused:
            return []

        if self.reranker is not None:
            pool = fused[: max(top_k * 4, 12)]
            return self.reranker.rerank(query, pool, top_k=top_k)

        return fused[:top_k]
