# 📄 RAG Document Q&A System

An intelligent document question-answering system built with Retrieval Augmented Generation (RAG). Upload PDFs or text files and ask questions — the AI finds relevant sections and generates accurate answers.

## 🎯 Features

- **Document Upload**: Support for PDF and TXT files
- **Intelligent Q&A**: Ask questions, get answers with source citations
- **Document Summarization**: Auto-summarize uploaded documents
- **Multi-document Search**: Search across all uploaded documents
- **Dual LLM Support**: HuggingFace (free) or OpenAI

## 🏗️ Architecture
```
User Question
  ↓
Embed query (Sentence-Transformers)
  ↓
Search vector database (FAISS)
  ↓
Retrieve relevant chunks
  ↓
Build prompt with context
  ↓
Generate answer (HuggingFace / OpenAI)
  ↓
Return answer + source citations
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend API | FastAPI |
| Frontend | Streamlit |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| Vector Database | FAISS |
| LLM | HuggingFace (flan-t5-large) / OpenAI |
| PDF Processing | pdfplumber |
| Containerization | Docker + Docker Compose |

## 🚀 Quick Start

### With Docker (Recommended)
```bash
git clone https://github.com/yarramasu0911/rag-document-qa.git
cd rag-document-qa
docker-compose up --build
```
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000/docs

### Without Docker
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Terminal 1: Start backend
uvicorn app.main:app --reload

# Terminal 2: Start frontend
streamlit run streamlit_app/app.py
```

## 📁 Project Structure
```
rag-document-qa/
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI endpoints
│   ├── document_processor.py  # PDF parsing, text chunking
│   ├── embeddings.py          # Sentence-Transformer embeddings
│   ├── vector_store.py        # FAISS vector search
│   ├── rag_chain.py           # RAG pipeline (retrieve + generate)
│   └── config.py              # Settings
├── streamlit_app/
│   └── app.py                 # Streamlit UI
├── tests/
│   ├── test_processor.py
│   ├── test_embeddings.py
│   └── test_rag.py
├── data/
│   └── sample_docs/
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload PDF/TXT document |
| POST | `/ask?question=...` | Ask a question |
| POST | `/summarize` | Summarize all documents |
| GET | `/documents` | List uploaded documents |
| POST | `/clear` | Clear all documents |
| GET | `/health` | Health check |

## 💡 How RAG Works

1. **Ingest**: Documents are split into chunks (~500 chars) and converted to vector embeddings
2. **Retrieve**: User question is embedded and compared against stored vectors using FAISS similarity search
3. **Generate**: Most relevant chunks are passed as context to the LLM, which generates an answer grounded in the document content

## 🔧 Configuration

Set in `.env` file:
```
LLM_PROVIDER=huggingface    # or "openai"
OPENAI_API_KEY=your_key     # only if using OpenAI
```

## 👤 Author

**Prasanth Yarramasu**