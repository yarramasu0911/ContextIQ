# tests/test_rag.py
import sys

sys.path.append(".")

from app.document_processor import process_document
from app.embeddings import EmbeddingModel
from app.vector_store import VectorStore
from app.rag_chain import RAGChain

# Setup
chunks = process_document("data/sample_docs/CaseFiles.pdf")
model = EmbeddingModel()
embeddings = model.embed_texts(chunks)

store = VectorStore(dimension=model.get_dimension())
store.add_documents(chunks, embeddings)

# Create RAG chain (free HuggingFace)
rag = RAGChain(embedding_model=model, vector_store=store, llm_provider="huggingface")

# Ask questions!
questions = [
    "What metaphors are discussed in this document?",
    "What is this document about?",
]

for q in questions:
    print(f"\nQuestion: {q}")
    result = rag.ask(q)
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])} chunks used")
