"""Two orthogonal things live here:

- TraceBuffer: a ring buffer of recent `/ask` invocations with per-stage
  internals (retrieved candidates, fused list, reranked list, prompt, grounding).
  Read via the admin-only /debug/trace endpoint.

- AuditLog: append-only JSONL record of every `/ask` (who, when, question,
  which documents were retrieved, the final answer). For HR/compliance.
"""

from __future__ import annotations

import collections
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class TraceBuffer:
    def __init__(self, capacity: int = 64):
        self._buf: collections.deque = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()

    def record(self, entry: dict[str, Any]) -> str:
        entry = dict(entry)
        entry.setdefault("id", uuid.uuid4().hex[:10])
        entry.setdefault("ts", time.time())
        with self._lock:
            self._buf.appendleft(entry)
        return entry["id"]

    def recent(self, n: int = 20) -> list[dict]:
        with self._lock:
            return list(self._buf)[:n]

    def find(self, trace_id: str) -> dict | None:
        with self._lock:
            for e in self._buf:
                if e.get("id") == trace_id:
                    return e
        return None


class AuditLog:
    def __init__(self, path: str = "data/audit.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, entry: dict[str, Any]):
        entry = dict(entry)
        entry.setdefault("ts", time.time())
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    def tail(self, n: int = 100) -> list[dict]:
        if not self.path.exists():
            return []
        with self._lock:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        tail = lines[-n:]
        out = []
        for ln in tail:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return out
