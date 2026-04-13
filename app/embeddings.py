from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
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
