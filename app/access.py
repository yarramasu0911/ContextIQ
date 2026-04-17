"""Role/visibility access control helpers shared by the vector store, chunk
registry, and API layer. Keeps the "who can see what" rules in one place.
"""

from __future__ import annotations

from typing import Optional


def pinecone_filter(
    viewer_username: str, viewer_role: str
) -> Optional[dict]:
    """Build a Pinecone metadata filter for the viewer's access scope.

    admin -> sees everything (no filter).
    hr    -> sees public + team, plus own private docs.
    user  -> sees public, plus own private docs.
    """
    if viewer_role == "admin":
        return None
    if viewer_role == "hr":
        return {
            "$or": [
                {"visibility": {"$in": ["public", "team"]}},
                {"owner_username": {"$eq": viewer_username}},
            ]
        }
    # default / user
    return {
        "$or": [
            {"visibility": {"$eq": "public"}},
            {"owner_username": {"$eq": viewer_username}},
        ]
    }


def can_see(
    viewer_username: str,
    viewer_role: str,
    owner_username: str,
    visibility: str,
) -> bool:
    """In-memory visibility check (used by BM25 registry).

    Mirrors the Pinecone filter above but applied per-chunk in Python.
    """
    if viewer_role == "admin":
        return True
    if visibility == "public":
        return True
    if owner_username == viewer_username:
        return True
    if viewer_role == "hr" and visibility == "team":
        return True
    return False
