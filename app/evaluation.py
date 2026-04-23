"""Lightweight RAG evaluation harness.

Inspired by RAGAS but self-contained — no extra heavyweight dep required. Given
a gold set of questions with ground-truth answers and (optionally) the expected
supporting snippets, it scores a RAGChain on:

- context_recall: did retrieval surface the expected supporting text?
- context_precision: how much of what was retrieved is relevant?
- answer_similarity: cosine similarity between generated and reference answer.
- faithfulness: share of answer sentences supported by retrieved context
  (reuses `faithfulness.check_grounding`).

Run it as a script:

    python -m app.evaluation path/to/golden.json
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .faithfulness import check_grounding


@dataclass
class GoldenItem:
    question: str
    reference_answer: str
    reference_contexts: list[str]  # substrings that must appear in retrieval

    @classmethod
    def from_dict(cls, d: dict) -> "GoldenItem":
        return cls(
            question=d["question"],
            reference_answer=d.get("answer", ""),
            reference_contexts=d.get("contexts", []) or [],
        )


def _context_recall(
    retrieved_texts: list[str], expected_snippets: list[str]
) -> float:
    if not expected_snippets:
        return 1.0
    joined = " ".join(retrieved_texts).lower()
    hits = sum(1 for s in expected_snippets if s.lower() in joined)
    return hits / len(expected_snippets)


def _context_precision(
    retrieved_texts: list[str],
    expected_snippets: list[str],
    embedding_model,
    relevance_threshold: float = 0.5,
) -> float:
    if not retrieved_texts:
        return 0.0
    if not expected_snippets:
        # no gold snippets: fall back to embedding-similarity to reference answer
        return 1.0

    ret_emb = np.asarray(
        embedding_model.embed_texts(retrieved_texts), dtype=np.float32
    )
    exp_emb = np.asarray(
        embedding_model.embed_texts(expected_snippets), dtype=np.float32
    )
    sims = ret_emb @ exp_emb.T  # [n_retrieved, n_expected]
    best_per_chunk = sims.max(axis=1)
    relevant = (best_per_chunk >= relevance_threshold).sum()
    return float(relevant) / len(retrieved_texts)


def _answer_similarity(
    generated: str, reference: str, embedding_model
) -> float:
    if not generated or not reference:
        return 0.0
    embs = np.asarray(
        embedding_model.embed_texts([generated, reference]), dtype=np.float32
    )
    return float(embs[0] @ embs[1])


def evaluate(
    rag_chain,
    embedding_model,
    golden: list[GoldenItem],
    user_id: str = "default",
    verbose: bool = True,
) -> dict:
    per_item = []
    t0 = time.time()
    for i, item in enumerate(golden):
        start = time.time()
        result = rag_chain.ask(item.question, user_id=user_id)
        latency = time.time() - start

        sources = result.get("sources", []) or []
        retrieved_texts = [s.get("text", "") for s in sources]

        recall = _context_recall(retrieved_texts, item.reference_contexts)
        precision = _context_precision(
            retrieved_texts, item.reference_contexts, embedding_model
        )
        answer_sim = _answer_similarity(
            result.get("answer", ""), item.reference_answer, embedding_model
        )

        grounding = check_grounding(
            result.get("answer", ""),
            [{"text": s.get("text", ""), "document": s.get("document")} for s in sources],
            embedding_model,
        )

        row = {
            "question": item.question,
            "answer": result.get("answer", ""),
            "reference": item.reference_answer,
            "latency_s": round(latency, 3),
            "context_recall": round(recall, 3),
            "context_precision": round(precision, 3),
            "answer_similarity": round(answer_sim, 3),
            "faithfulness": round(grounding["grounding_score"], 3),
            "cached": bool(result.get("cached", False)),
        }
        per_item.append(row)
        if verbose:
            print(
                f"[{i + 1}/{len(golden)}] "
                f"recall={row['context_recall']:.2f} "
                f"prec={row['context_precision']:.2f} "
                f"sim={row['answer_similarity']:.2f} "
                f"faith={row['faithfulness']:.2f} "
                f"({row['latency_s']}s)"
            )

    total_time = time.time() - t0

    def _mean(key):
        vals = [r[key] for r in per_item]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    summary = {
        "n_items": len(per_item),
        "total_seconds": round(total_time, 2),
        "mean_context_recall": _mean("context_recall"),
        "mean_context_precision": _mean("context_precision"),
        "mean_answer_similarity": _mean("answer_similarity"),
        "mean_faithfulness": _mean("faithfulness"),
        "mean_latency_s": _mean("latency_s"),
        "cache_hit_rate": round(
            sum(1 for r in per_item if r["cached"]) / max(1, len(per_item)), 3
        ),
    }
    return {"summary": summary, "items": per_item}


def load_golden(path: str | Path) -> list[GoldenItem]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GoldenItem.from_dict(d) for d in data]


def _main(argv: list[str]):
    if len(argv) < 2:
        print("usage: python -m app.evaluation <golden.json> [output.json]")
        sys.exit(1)

    golden_path = argv[1]
    output_path = argv[2] if len(argv) > 2 else None

    from .embeddings import EmbeddingModel
    from .vector_store_pinecone import PineconeVectorStore
    from .rag_chain import RAGChain

    embedding_model = EmbeddingModel()
    vector_store = PineconeVectorStore()
    rag_chain = RAGChain(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm_provider="huggingface",
    )

    golden = load_golden(golden_path)
    report = evaluate(rag_chain, embedding_model, golden)

    print("\n=== SUMMARY ===")
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")

    if output_path:
        Path(output_path).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(f"\nWrote detailed report to {output_path}")


if __name__ == "__main__":
    _main(sys.argv)
