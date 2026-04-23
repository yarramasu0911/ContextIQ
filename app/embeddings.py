import os

from sentence_transformers import SentenceTransformer
import numpy as np

# Set EMBEDDING_MODEL to switch, e.g.:
#   intfloat/multilingual-e5-large   (multilingual, 1024-dim)
#   BAAI/bge-small-en-v1.5           (English, 384-dim, fast)
#   all-MiniLM-L6-v2                 (English, 384-dim, fastest)
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")


class EmbeddingModel:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or DEFAULT_MODEL
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )

        return embedding.astype("float32")[0]
