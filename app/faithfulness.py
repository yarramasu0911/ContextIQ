"""Grounding / faithfulness check for generated answers.

For each sentence in the answer, find the best-matching source chunk by cosine
similarity of embeddings. Sentences below the support threshold are flagged as
potential hallucinations so the UI can highlight them and the API can report
an overall grounding score.

This is the cheap, model-agnostic version of RAGAS `faithfulness` — good enough
to catch obvious invented facts without a second LLM call.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def check_grounding(
    answer: str,
    sources: list[dict],
    embedding_model,
    support_threshold: float = 0.55,
) -> dict:
    """Return per-sentence support scores and an overall grounding score.

    sources is a list of dicts with at least a "text" field. The "document"
    field (filename) is echoed back on each sentence's best match so the UI
    can render inline citations like "[doc.pdf]".
    """
    sentences = split_sentences(answer)
    if not sentences or not sources:
        return {
            "sentences": [
                {
                    "text": s,
                    "supported": False,
                    "score": 0.0,
                    "best_source": None,
                }
                for s in sentences
            ],
            "grounding_score": 0.0,
            "unsupported_count": len(sentences),
        }

    source_texts = [s.get("text", "") for s in sources]
    source_embs = np.asarray(
        embedding_model.embed_texts(source_texts), dtype=np.float32
    )
    sent_embs = np.asarray(
        embedding_model.embed_texts(sentences), dtype=np.float32
    )

    # embeddings are L2-normalized by EmbeddingModel, so dot == cosine
    sims = sent_embs @ source_embs.T  # [n_sentences, n_sources]

    annotated = []
    unsupported = 0
    for i, sent in enumerate(sentences):
        best_idx = int(np.argmax(sims[i]))
        best_score = float(sims[i, best_idx])
        supported = best_score >= support_threshold
        if not supported:
            unsupported += 1
        annotated.append(
            {
                "text": sent,
                "supported": supported,
                "score": best_score,
                "best_source": sources[best_idx].get("document")
                or sources[best_idx].get("filename"),
            }
        )

    grounding_score = 1.0 - (unsupported / len(sentences))
    return {
        "sentences": annotated,
        "grounding_score": grounding_score,
        "unsupported_count": unsupported,
    }


def annotate_with_citations(grounding: dict) -> Optional[str]:
    """Render the annotated answer with inline [source] citations.

    Returns None if nothing to annotate.
    """
    sents = grounding.get("sentences") or []
    if not sents:
        return None
    out_parts = []
    for s in sents:
        if s["supported"] and s["best_source"]:
            out_parts.append(f"{s['text']} [{s['best_source']}]")
        elif not s["supported"]:
            out_parts.append(f"{s['text']} [unverified]")
        else:
            out_parts.append(s["text"])
    return " ".join(out_parts)
