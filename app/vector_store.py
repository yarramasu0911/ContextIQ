import numpy as np
import faiss


class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

        # Use inner product for normalized embeddings
        self.index = faiss.IndexFlatIP(dimension)

        # Store full chunk documents, not just strings
        self.documents = []

    def add_documents(self, chunk_docs: list[dict], embeddings: np.ndarray):
        """
        chunk_docs format:
        [
            {
                "content": "...",
                "metadata": {...}
            }
        ]
        """
        if not chunk_docs:
            return

        vectors = np.array(embeddings, dtype="float32")

        if len(chunk_docs) != len(vectors):
            raise ValueError("Number of chunk docs must match number of embeddings")

        self.index.add(vectors)
        self.documents.extend(chunk_docs)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        if self.index.ntotal == 0:
            return []

        query_vector = np.array([query_embedding], dtype="float32")

        # ask for more candidates so we can deduplicate
        scores, indices = self.index.search(query_vector, top_k * 3)

        results = []
        seen = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            doc = self.documents[idx]

            key = (doc["metadata"].get("file_name"), doc["metadata"].get("chunk_index"))

            if key in seen:
                continue

            seen.add(key)

            results.append(
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": float(score),
                }
            )

            if len(results) == top_k:
                break

        return results
