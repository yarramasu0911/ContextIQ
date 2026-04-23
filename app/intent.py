"""Zero-shot intent classifier for incoming questions.

Classifies each question into one of: casual, summary, factoid, comparison,
list, other. Cheap — no LLM call, just cosine against a fixed set of template
sentences per class. Used to replace the old keyword-matching summary shortcut
and to pick prompt templates per intent.
"""

from __future__ import annotations

import numpy as np


_TEMPLATES: dict[str, list[str]] = {
    "casual": [
        "Hello",
        "Thanks",
        "Goodbye",
        "Who are you",
        "What can you do",
        "How are you",
    ],
    "summary": [
        "Summarize this document",
        "Give me an overview of the document",
        "What are the main points?",
        "Provide a brief summary of the contents",
        "What is this document about overall?",
    ],
    "factoid": [
        "What is the date of the meeting?",
        "Who is the author of this report?",
        "How much does it cost?",
        "What is the deadline?",
        "Where is the event held?",
    ],
    "comparison": [
        "Compare A and B",
        "What is the difference between X and Y?",
        "How does A differ from B?",
        "Which is better, A or B?",
    ],
    "list": [
        "List all the requirements",
        "Enumerate the steps",
        "What are the items included?",
        "Name the parties involved",
    ],
}


class IntentClassifier:
    def __init__(self, embedding_model, threshold: float = 0.45):
        self.embedding_model = embedding_model
        self.threshold = threshold
        # [n_classes, n_templates_per_class, dim] flattened
        self._class_names: list[str] = []
        self._class_embs: np.ndarray | None = None
        self._build()

    def _build(self):
        all_texts: list[str] = []
        class_of: list[str] = []
        for cls, examples in _TEMPLATES.items():
            for ex in examples:
                all_texts.append(ex)
                class_of.append(cls)
        embs = np.asarray(
            self.embedding_model.embed_texts(all_texts), dtype=np.float32
        )
        self._class_names = class_of
        self._class_embs = embs

    def classify(self, question: str) -> tuple[str, float]:
        """Return (intent, confidence). Falls back to 'other' below threshold."""
        if self._class_embs is None:
            return "other", 0.0
        q = np.asarray(
            self.embedding_model.embed_query(question), dtype=np.float32
        )
        sims = self._class_embs @ q
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score < self.threshold:
            return "other", score
        return self._class_names[best], score


INTENT_PROMPT_HINTS = {
    "factoid": "Answer with the specific fact requested, in a single concise sentence.",
    "comparison": "Answer by comparing the two items point by point using only the sources.",
    "list": "Answer as a numbered or bulleted list, one item per line, using only the sources.",
    "summary": "Answer with a concise 2-3 sentence summary.",
    "other": "Answer concisely in 2-3 sentences.",
}
