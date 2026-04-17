from pinecone import Pinecone
import os
import uuid

from .access import pinecone_filter


class PineconeVectorStore:
    def __init__(self, dimension: int = 768):
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index(os.getenv("PINECONE_INDEX", "contextiq"))
        self.dimension = dimension

    def add_documents(
        self,
        chunks: list,
        embeddings,
        filename: str = "unknown",
        owner_username: str = "default",
        owner_role: str = "user",
        visibility: str = "private",
    ):
        vectors = []
        for i in range(len(chunks)):
            chunk = chunks[i]
            embedding = embeddings[i]

            if isinstance(chunk, dict):
                text = chunk.get("content", "no content")
            else:
                text = chunk

            text = str(text)[:1000]

            if hasattr(embedding, "tolist"):
                emb_list = embedding.tolist()
            else:
                emb_list = list(embedding)

            vectors.append(
                {
                    "id": str(uuid.uuid4()),
                    "values": emb_list,
                    "metadata": {
                        "text": text,
                        "filename": str(filename),
                        # legacy field, kept for back-compat
                        "user_id": str(owner_username),
                        "owner_username": str(owner_username),
                        "owner_role": str(owner_role),
                        "visibility": str(visibility),
                    },
                }
            )

        for i in range(0, len(vectors), 100):
            self.index.upsert(vectors=vectors[i : i + 100])

    def search(
        self,
        query_embedding,
        top_k: int = 3,
        viewer_username: str = "default",
        viewer_role: str = "admin",
    ) -> list:
        if hasattr(query_embedding, "tolist"):
            query_list = query_embedding.tolist()
        else:
            query_list = list(query_embedding)

        flt = pinecone_filter(viewer_username, viewer_role)
        kwargs = {
            "vector": query_list,
            "top_k": top_k,
            "include_metadata": True,
        }
        if flt is not None:
            kwargs["filter"] = flt

        results = self.index.query(**kwargs)

        return [
            (
                match.metadata["text"],
                match.score,
                match.metadata.get("filename", "unknown"),
            )
            for match in results.matches
        ]

    def _list_matches(self, viewer_username: str, viewer_role: str, top_k: int = 10000):
        flt = pinecone_filter(viewer_username, viewer_role)
        kwargs = {
            "vector": [0.0] * self.dimension,
            "top_k": top_k,
            "include_metadata": True,
        }
        if flt is not None:
            kwargs["filter"] = flt
        return self.index.query(**kwargs).matches

    def get_document_list(
        self, viewer_username: str = "default", viewer_role: str = "admin"
    ) -> list:
        matches = self._list_matches(viewer_username, viewer_role, top_k=1000)
        seen = {}
        for m in matches:
            fn = m.metadata.get("filename", "unknown")
            if fn not in seen:
                seen[fn] = {
                    "filename": fn,
                    "owner_username": m.metadata.get("owner_username", "?"),
                    "owner_role": m.metadata.get("owner_role", "?"),
                    "visibility": m.metadata.get("visibility", "?"),
                }
        return list(seen.values())

    def clear(
        self,
        viewer_username: str = "default",
        viewer_role: str = "admin",
        scope: str = "mine",
    ):
        """Clear documents.

        scope='mine' deletes only docs owned by viewer_username.
        scope='all'  deletes every doc the viewer can see (admins only).
        """
        if scope == "all" and viewer_role == "admin":
            matches = self._list_matches(viewer_username, "admin", top_k=10000)
        else:
            # own-only: filter by owner_username
            matches = self.index.query(
                vector=[0.0] * self.dimension,
                top_k=10000,
                include_metadata=True,
                filter={"owner_username": {"$eq": viewer_username}},
            ).matches

        ids = [m.id for m in matches]
        if ids:
            # delete in batches of 1000
            for i in range(0, len(ids), 1000):
                self.index.delete(ids=ids[i : i + 1000])

    def get_total_vectors(
        self, viewer_username: str = "default", viewer_role: str = "admin"
    ) -> int:
        return len(self._list_matches(viewer_username, viewer_role, top_k=10000))
