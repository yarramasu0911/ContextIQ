import streamlit as st

st.set_page_config(page_title="ContextIQ", page_icon="📄", layout="wide")


import sys
import os
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.document_processor import process_document
from app.embeddings import EmbeddingModel
from app.vector_store_pinecone import PineconeVectorStore
from app.rag_chain import RAGChain
from app.auth import register_user, authenticate_user

# Load environment
from dotenv import load_dotenv

load_dotenv()

from streamlit_cookies_controller import CookieController

cookies = CookieController()


@st.cache_resource
def load_models():
    embedding_model = EmbeddingModel()
    vector_store = PineconeVectorStore()
    rag_chain = RAGChain(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm_provider="huggingface",
    )
    return embedding_model, vector_store, rag_chain


def check_login():
    """Check if user is logged in via session or cookie."""
    if st.session_state.get("authenticated"):
        return True

    # Wait for cookies to load
    time.sleep(0.5)

    # Check cookie
    saved_user = cookies.get("contextiq_user")
    if saved_user:
        st.session_state["authenticated"] = True
        st.session_state["username"] = saved_user
        return True

    return False


def login_page():
    """Show login/register page."""
    st.title("📄 ContextIQ - Document Q&A")
    st.write("Login or register to start asking questions about your documents.")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                cookies.set("contextiq_user", username)
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        new_username = st.text_input("Choose Username", key="reg_user")
        new_password = st.text_input("Choose Password", type="password", key="reg_pass")
        confirm_password = st.text_input(
            "Confirm Password", type="password", key="reg_confirm"
        )
        if st.button("Register"):
            if new_password != confirm_password:
                st.error("Passwords don't match")
            elif len(new_password) < 4:
                st.error("Password must be at least 4 characters")
            elif register_user(new_username, new_password):
                st.success("Registered! Please login.")
            else:
                st.error("Username already exists")


def main_app():
    """Main application after login."""
    username = st.session_state["username"]
    embedding_model, vector_store, rag_chain = load_models()

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.title("📄 ContextIQ - Document Q&A")

    # ---- Sidebar ----
    with st.sidebar:
        st.write(f"👤 Logged in as: **{username}**")
        if st.button("Logout"):
            st.session_state["authenticated"] = False
            st.session_state["username"] = None
            st.query_params.clear()
            st.rerun()

        st.divider()
        st.header("📁 Upload Documents")
        uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

        if uploaded_file and st.button("Process Document"):
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                chunks = process_document(
                    tmp_path, use_semantic=True, embedding_model=embedding_model.model
                )

                chunk_texts = [chunk["content"] for chunk in chunks]
                embeddings = embedding_model.embed_texts(chunk_texts)

                vector_store.add_documents(
                    chunks, embeddings, filename=uploaded_file.name, user_id=username
                )
                os.unlink(tmp_path)

            st.success(f"✅ '{uploaded_file.name}' processed!\nChunks: {len(chunks)}")

        st.divider()
        if st.button("🗑️ Clear My Documents"):
            vector_store.clear(user_id=username)
            st.success("All your documents cleared!")

        st.divider()
        docs = vector_store.get_document_list(user_id=username)
        total = vector_store.get_total_vectors(user_id=username)
        st.write(f"📊 **Your documents:** {len(docs)}")
        st.write(f"📊 **Total chunks:** {total}")
        for doc in docs:
            st.write(f"  • {doc}")

        st.divider()
        if st.button("📋 Summarize Documents"):
            total = vector_store.get_total_vectors(user_id=username)
            if total == 0:
                st.error("No documents uploaded yet.")
            else:
                with st.spinner("Generating summary..."):
                    result = rag_chain.summarize(user_id=username)
                st.session_state["chat_history"].append(
                    {"role": "user", "content": "Summarize the documents"}
                )
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": result["answer"], "sources": []}
                )
                st.rerun()

        if st.button("🧹 Clear Chat"):
            st.session_state["chat_history"] = []
            st.rerun()

    # ---- Main Area: Chat Interface ----
    st.subheader("💬 Chat with your Documents")

    # Display chat history
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
                if msg.get("sources"):
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(msg["sources"]):
                            st.markdown(
                                f"**Source {i + 1}** from _{source['document']}_"
                            )
                            st.info(source["text"])

    # Chat input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        total = vector_store.get_total_vectors(user_id=username)
        if total == 0:
            st.chat_message("assistant").write(
                "No documents uploaded yet. Please upload a document first."
            )
        else:
            # Show user message
            st.session_state["chat_history"].append(
                {"role": "user", "content": question}
            )
            st.chat_message("user").write(question)

            # Get answer
            with st.spinner("Finding answer..."):
                result = rag_chain.ask(
                    question,
                    user_id=username,
                    chat_history=st.session_state["chat_history"],
                )

            # Show and store assistant message
            st.session_state["chat_history"].append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                }
            )

            with st.chat_message("assistant"):
                st.write(result["answer"])
                if result.get("sources"):
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(result["sources"]):
                            st.markdown(
                                f"**Source {i + 1}** from _{source['document']}_"
                            )
                            st.info(source["text"])


def main():

    if check_login():
        main_app()
    else:
        login_page()


if __name__ == "__main__":
    main()
