"""Query transformations that improve retrieval:

- rewrite_with_history: resolve pronouns / follow-ups using chat history so the
  standalone query is retrievable on its own.
- hyde: generate a hypothetical answer and embed that instead of (or alongside)
  the raw question. Great for vague queries.
- multi_query: paraphrase the question N ways and retrieve for each.

All transforms take a callable `llm_call(prompt: str) -> str` so they work with
either the HuggingFace or OpenAI backend without pulling in either directly.
"""

from __future__ import annotations

import re
from typing import Callable, Optional


LLM = Callable[[str], str]


def _format_history(chat_history: Optional[list[dict]], max_turns: int = 3) -> str:
    if not chat_history:
        return ""
    recent = chat_history[-(max_turns * 2) :]
    lines = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def rewrite_with_history(
    question: str, chat_history: Optional[list[dict]], llm: LLM
) -> str:
    """Rewrite a follow-up question into a standalone form.

    Returns the original question unchanged if there is no history or the LLM
    returns an empty/degenerate rewrite.
    """
    history = _format_history(chat_history)
    if not history:
        return question

    prompt = (
        "Given the conversation history and a follow-up question, rewrite the "
        "follow-up as a standalone question that can be understood without the "
        "history. Preserve the user's intent. Return only the rewritten "
        "question, no preamble.\n\n"
        f"Conversation:\n{history}\n\n"
        f"Follow-up: {question}\n\n"
        "Standalone question:"
    )

    try:
        rewritten = (llm(prompt) or "").strip()
    except Exception:
        return question

    # guard against empty, too-long, or obviously bad outputs
    if not rewritten or len(rewritten) > 400 or len(rewritten) < 3:
        return question
    return rewritten


def hyde(question: str, llm: LLM) -> Optional[str]:
    """Generate a short hypothetical answer to embed instead of the question.

    Returns None on failure so callers can fall back to plain retrieval.
    """
    prompt = (
        "Write a short, factual-sounding passage (2-3 sentences) that would "
        "plausibly answer the question below. Do not say you are unsure — just "
        "write the passage as if it were from a document. This text will be "
        "used for semantic search, not shown to the user.\n\n"
        f"Question: {question}\n\n"
        "Passage:"
    )
    try:
        out = (llm(prompt) or "").strip()
    except Exception:
        return None
    if not out or len(out) < 10:
        return None
    return out


_QUERY_LINE = re.compile(r"^\s*(?:\d+[\.\)]\s*|[-*]\s*)?(.+?)\s*$")


def multi_query(question: str, llm: LLM, n: int = 3) -> list[str]:
    """Generate N paraphrases of the question. Always includes the original."""
    prompt = (
        f"Generate {n} alternative phrasings of the question below. Each should "
        "preserve the original intent but use different wording or emphasis, to "
        "improve document retrieval coverage. Return one per line, no "
        "numbering, no explanations.\n\n"
        f"Question: {question}\n\n"
        "Alternatives:"
    )
    try:
        raw = llm(prompt) or ""
    except Exception:
        return [question]

    variants = []
    for line in raw.splitlines():
        m = _QUERY_LINE.match(line)
        if not m:
            continue
        v = m.group(1).strip().strip('"').strip("'")
        if 3 < len(v) < 300 and v.lower() != question.lower():
            variants.append(v)

    return [question] + variants[:n]
