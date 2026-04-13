from pinecone import Pinecone
import os
import uuid


class PineconeVectorStore:
    def __init__(self,dimension: int = 768):
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index(os.getenv("PINECONE_INDEX", "contextiq"))
        self.dimension = dimension

    def add_documents(self, chunks: list, embeddings, filename: str = "unknown", user_id: str = "default"):
        vectors = []

        for i in range(len(chunks)):
            chunk = chunks[i]
            embedding = embeddings[i]

            if isinstance(chunk, dict):
                text = chunk.get("content", "no content")
            else:
                text = chunk

            text = str(text)[:1000]

            if hasattr(embedding, 'tolist'):
                emb_list = embedding.tolist()
            else:
                emb_list = list(embedding)

            vectors.append({
                "id": str(uuid.uuid4()),
                "values": emb_list,
                "metadata": {
                    "text": text,
                    "filename": str(filename),
                    "user_id": str(user_id)
                }
            })

        for i in range(0, len(vectors), 100):
            self.index.upsert(vectors=vectors[i:i + 100])

    def search(self, query_embedding, top_k: int = 3, user_id: str = "default") -> list:
        if hasattr(query_embedding, 'tolist'):
            query_list = query_embedding.tolist()
        else:
            query_list = list(query_embedding)

        results = self.index.query(
            vector=query_list,
            top_k=top_k,
            include_metadata=True,
            filter={"user_id": {"$eq": user_id}}
        )

        return [
            (match.metadata["text"], match.score, match.metadata.get("filename", "unknown"))
            for match in results.matches
        ]

    def get_document_list(self, user_id: str = "default") -> list:
        results = self.index.query(
            vector=[0.0] * self.dimension,
            top_k=1000,
            include_metadata=True,
            filter={"user_id": {"$eq": user_id}}
        )
        filenames = set(m.metadata.get("filename", "unknown") for m in results.matches)
        return list(filenames)

    def clear(self, user_id: str = "default"):
        results = self.index.query(
            vector=[0.0] * self.dimension,
            top_k=10000,
            include_metadata=True,
            filter={"user_id": {"$eq": user_id}}
        )
        ids = [m.id for m in results.matches]
        if ids:
            self.index.delete(ids=ids)

    def get_total_vectors(self, user_id: str = "default") -> int:
        results = self.index.query(
            vector=[0.0] * self.dimension,
            top_k=10000,
            filter={"user_id": {"$eq": user_id}}
        )
        return len(results.matches)