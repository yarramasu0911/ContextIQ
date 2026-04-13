# tests/test_vector_store.py
import sys

sys.path.append(".")

from app.document_processor import process_document
from app.embeddings import EmbeddingModel
from app.vector_store import PineconeVectorStore

# Process document
chunks = process_document("data/sample_docs/CaseFiles.pdf")

# Create embeddings
model = EmbeddingModel()
embeddings = model.embed_texts(chunks)

# Store in vector store
store = PineconeVectorStore(dimension=384)
store.add_documents(chunks, embeddings)

print(f"Stored {store.index.ntotal} vectors")

# Search
query = "What metaphors are discussed?"
query_embedding = model.embed_query(query)
results = store.search(query_embedding, top_k=3)

print(f"\nQuery: {query}")
print(f"\nTop 3 results:")
for i, (chunk, score) in enumerate(results):
    print(f"\n--- Result {i + 1} (distance: {score:.4f}) ---")
    print(chunk[:200])
