import streamlit as st
import requests

API_URL = "http://localhost:8000"


def upload_document(file):
    """Upload document to FastAPI backend."""
    files = {"file": (file.name, file.getvalue(), file.type)}
    response = requests.post(f"{API_URL}/upload", files=files)
    return response.json()


def ask_question(question: str):
    """Send question to FastAPI backend."""
    response = requests.post(f"{API_URL}/ask", params={"question": question})
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.json().get("error", "Unknown error")}


def summarize_document():
    """Get document summary from backend."""
    response = requests.post(f"{API_URL}/summarize")
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.json().get("error", "Unknown error")}


def main():
    st.set_page_config(page_title="HelpDesk Assist", page_icon="📄", layout="wide")
    st.title("📄 HelpDesk Assist")
    st.write("Upload documents and ask questions about them!")

    # ---- Sidebar: Document Upload ----
    with st.sidebar:
        st.header("📁 Upload Documents")
        uploaded_file = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            help="Upload a document to ask questions about",
        )

        if uploaded_file and st.button("Process Document"):
            with st.spinner("Processing document..."):
                result = upload_document(uploaded_file)

            if "error" in result:
                st.error(result["error"])
            else:
                st.success(
                    f"✅ '{uploaded_file.name}' processed!\n\n"
                    f"Chunks: {result['chunks']}\n\n"
                    f"Total stored: {result['total_documents_stored']}"
                )

        st.divider()
        st.header("ℹ️ How it works")
        st.write(
            "1. Upload a PDF or TXT document\n"
            "2. Ask questions about the content\n"
            "3. AI finds relevant sections and answers"
        )

    # ---- Main Area: Q&A ----
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
            with st.spinner("Finding answer..."):
                result = ask_question(question)

            if "error" in result:
                st.error(result["error"])
            else:
                st.subheader("📝 Answer:")
                st.write(result["answer"])

                with st.expander("📚 View Sources"):
                    for i, (source, score) in enumerate(
                        zip(result["sources"], result["scores"])
                    ):
                        st.markdown(
                            f"**Source {i + 1}** (relevance: {1 / (1 + score):.2%})"
                        )
                        st.info(source)

        elif ask_clicked and not question:
            st.warning("Please enter a question.")

        if summarize_clicked:
            with st.spinner("Generating summary..."):
                result = summarize_document()

            if "error" in result:
                st.error(result["error"])
            else:
                st.subheader("📋 Summary:")
                st.write(result["answer"])

    with col2:
        st.subheader("💡 Example Questions")
        example_questions = [
            "What is this document about?",
            "What are the key points?",
            "What metaphors are discussed?",
        ]
        for eq in example_questions:
            if st.button(eq, key=eq):
                st.session_state["question"] = eq


if __name__ == "__main__":
    main()
