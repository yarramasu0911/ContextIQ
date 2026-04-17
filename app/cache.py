"""Semantic Q&A cache.

Skip the full retrieve+generate pipeline when a new question is semantically
close to one already answered. Invalidated on document upload / clear so
answers never get served from a cache that no longer matches the corpus.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class _Entry:
    question: str
    embedding: np.ndarray
    answer: dict
    created_at: float = field(default_factory=time.time)
    hits: int = 0


class SemanticCache:
    def __init__(
        self,
        embedding_model,
        similarity_threshold: float = 0.95,
        max_entries: int = 256,
    ):
        self.embedding_model = embedding_model
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        # user_id -> list[_Entry]
        self._store: dict[str, list[_Entry]] = {}

    def lookup(self, question: str, user_id: str = "default") -> Optional[dict]:
        bucket = self._store.get(user_id)
        if not bucket:
            return None

        q_emb = self.embedding_model.embed_query(question)
        q_emb = np.asarray(q_emb, dtype=np.float32)

        best_sim = -1.0
        best_entry: Optional[_Entry] = None
        for entry in bucket:
            sim = float(np.dot(q_emb, entry.embedding))
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry is not None and best_sim >= self.threshold:
            best_entry.hits += 1
            answer = dict(best_entry.answer)
            answer["cached"] = True
            answer["cache_similarity"] = best_sim
            answer["cache_matched_question"] = best_entry.question
            return answer
        return None

    def store(self, question: str, answer: dict, user_id: str = "default"):
        if not answer or not answer.get("answer"):
            return
        bucket = self._store.setdefault(user_id, [])
        emb = np.asarray(
            self.embedding_model.embed_query(question), dtype=np.float32
        )
        bucket.append(_Entry(question=question, embedding=emb, answer=answer))

        # LRU-ish eviction: drop least-recent when over capacity
        if len(bucket) > self.max_entries:
            bucket.sort(key=lambda e: (e.hits, e.created_at))
            del bucket[: len(bucket) - self.max_entries]

    def invalidate(self, user_id: Optional[str] = None):
        if user_id is None:
            self._store.clear()
        else:
            self._store.pop(user_id, None)

    def stats(self, user_id: str = "default") -> dict:
        bucket = self._store.get(user_id, [])
        return {
            "entries": len(bucket),
            "total_hits": sum(e.hits for e in bucket),
        }
