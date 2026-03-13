from PyPDF2 import PdfReader
from docx import Document
import unicodedata
import re
import os
from bs4 import BeautifulSoup
import json
from striprtf.striprtf import rtf_to_text
import csv
from datetime import datetime


def clean_text(text: str) -> str:
    if not text:
        return ""

    # normalize unicode (fixes Word/PDF weird chars)
    text = unicodedata.normalize("NFKC", text)

    # remove invisible characters
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)

    # fix broken words from PDF line breaks
    text = re.sub(r"-\n", "", text)

    # convert single newlines inside sentences to space
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # reduce multiple newlines
    text = re.sub(r"\n{2,}", "\n", text)

    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_txt(file_path: str) -> str:
    """Read text from a TXT file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])


def extract_from_html(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return soup.get_text(separator="\n")


def extract_from_json(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    return json.dumps(data, indent=2)


def extract_from_rtf(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return rtf_to_text(f.read())


def extract_from_csv(file_path: str) -> str:
    rows = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(" | ".join(row))
    return "\n".join(rows)


def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> list:

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    step = chunk_size - overlap
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # try not to cut in the middle of a word
        if end < text_length:
            space_pos = text.rfind(" ", start, end)
            if space_pos > start:
                end = space_pos

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def ingest_document(file_path: str) -> list:
    """Main function: read file and return chunks."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext in [".txt", ".md"]:
        text = extract_text_from_txt(file_path)
    elif ext == ".docx":
        text = extract_from_docx(file_path)
    elif ext == ".csv":
        text = extract_from_csv(file_path)
    elif ext in [".html", ".htm"]:
        text = extract_from_html(file_path)
    elif ext == ".json":
        text = extract_from_json(file_path)
    elif ext == ".rtf":
        text = extract_from_rtf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    text = clean_text(text)
    return {
        "content": text,
        "metadata": {
            "file_name": os.path.basename(file_path),
            "file_type": ext,
            "source": "upload",
            "ingested_at": datetime.now().isoformat(),
        },
    }


def process_document(file_path: str, chunk_size: int = 500, overlap: int = 100) -> list:

    document = ingest_document(file_path)

    chunks = split_into_chunks(
        document["content"], chunk_size=chunk_size, overlap=overlap
    )

    chunk_docs = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        chunk_docs.append(
            {
                "content": chunk,
                "metadata": {
                    **document["metadata"],
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                },
            }
        )

    return chunk_docs
