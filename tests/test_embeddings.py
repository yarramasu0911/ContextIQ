import sys

sys.path.append(".")

from app.document_processor import process_document
from app.embeddings import EmbeddingModel

# Load chunks
chunks = process_document("data/sample_docs/CaseFiles.pdf")

# Create embeddings
model = EmbeddingModel()
embeddings = model.embed_texts(chunks)

print(f"Chunks: {len(chunks)}")
print(f"Embeddings: {len(embeddings)}")
print(f"Each embedding dimension: {len(embeddings[0])}")

# Test query
query_embedding = model.embed_query("What is this document about?")
print(f"\nQuery embedding dimension: {len(query_embedding)}")
print(f"First 5 values: {query_embedding[:5]}")
