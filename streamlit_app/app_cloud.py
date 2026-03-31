import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.document_processor import process_document
from app.embeddings import EmbeddingModel
from app.vector_store import VectorStore
from app.rag_chain import RAGChain
import tempfile


# Cache models so they load only once
@st.cache_resource
def load_models():
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(dimension=384)
    rag_chain = RAGChain(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm_provider="huggingface",
    )
    return embedding_model, vector_store, rag_chain


def main():
    st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")
    st.title("📄 RAG Document Q&A")
    st.write("Upload documents and ask questions about them!")

    # Load models
    embedding_model, vector_store, rag_chain = load_models()

    # ---- Sidebar: Upload ----
    with st.sidebar:
        st.header("📁 Upload Documents")
        uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

        if uploaded_file and st.button("Process Document"):
            with st.spinner("Processing document..."):
                # Save to temp file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                # Process
                chunks = process_document(tmp_path)
                embeddings = embedding_model.embed_texts(chunks)
                vector_store.add_documents(
                    chunks, embeddings, filename=uploaded_file.name
                )

                # Cleanup
                os.unlink(tmp_path)

            st.success(
                f"✅ '{uploaded_file.name}' processed!\n\nChunks: {len(chunks)}\n\nTotal stored: {vector_store.index.ntotal}"
            )

        st.divider()
        if st.button("🗑️ Clear All Documents"):
            vector_store.clear()
            st.success("All documents cleared!")

        st.divider()
        st.header("ℹ️ How it works")
        st.write(
            "1. Upload a PDF or TXT document\n2. Ask questions about the content\n3. AI finds relevant sections and answers"
        )

        if vector_store.index.ntotal > 0:
            st.divider()
            st.write(
                f"📊 **Documents stored:** {len(vector_store.get_document_list())}"
            )
            st.write(f"📊 **Total chunks:** {vector_store.index.ntotal}")
            for doc in vector_store.get_document_list():
                st.write(f"  • {doc}")

    # ---- Main Area ----
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Ask a Question")
        question = st.text_input(
            "Enter your question:", placeholder="What is this document about?"
        )

        btn_col1, btn_col2 = st.columns(2)
        ask_clicked = btn_col1.button("🔍 Ask", use_container_width=True)
        summarize_clicked = btn_col2.button("📋 Summarize", use_container_width=True)

        if ask_clicked and question:
            if vector_store.index.ntotal == 0:
                st.error("No documents uploaded yet. Upload a document first.")
            else:
                with st.spinner("Finding answer..."):
                    result = rag_chain.ask(question)

                st.subheader("📝 Answer:")
                st.write(result["answer"])

                with st.expander("📚 View Sources"):
                    for i, source in enumerate(result["sources"]):
                        if isinstance(source, dict):
                            st.markdown(
                                f"**Source {i + 1}** from _{source.get('document', 'unknown')}_"
                            )
                            st.info(source.get("text", ""))
                        else:
                            st.markdown(f"**Source {i + 1}**")
                            st.info(source)

        elif ask_clicked and not question:
            st.warning("Please enter a question.")

        if summarize_clicked:
            if vector_store.index.ntotal == 0:
                st.error("No documents uploaded yet.")
            else:
                with st.spinner("Generating summary..."):
                    result = rag_chain.summarize()
                st.subheader("📋 Summary:")
                st.write(result["answer"])

    with col2:
        st.subheader("💡 Example Questions")
        examples = [
            "What is this document about?",
            "What are the key points?",
            "What conclusions are drawn?",
        ]
        for eq in examples:
            if st.button(eq, key=eq):
                st.session_state["question"] = eq


if __name__ == "__main__":
    main()
