import sys

sys.path.append(".")
from app.document_processor import process_document

chunks = process_document("data/sample_docs/CaseFiles.pdf")

print(f"Number of chunks: {len(chunks)}")
print(f"\nFirst chunk ({len(chunks[0])} chars):")
print(chunks[0])
print(f"\nLast chunk ({len(chunks[-1])} chars):")
print(chunks[-1])
