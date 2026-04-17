import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

SUPPORTED_TYPES = [
    "pdf", "txt", "md", "log",
    "docx", "rtf",
    "csv", "tsv",
    "xlsx", "xls",
    "pptx", "ppt",
    "html", "htm", "xml",
    "json", "yaml", "yml",
]


# ------------------------------------------------------------------ API layer
def _auth_headers():
    if not st.session_state.get("authenticated"):
        return {}
    return {
        "X-Username": st.session_state["username"],
        "X-Password": st.session_state["password"],
    }


def api_login(username: str, password: str):
    r = requests.post(
        f"{API_URL}/auth/login",
        data={"username": username, "password": password},
    )
    return r.status_code, r.json() if r.headers.get("content-type", "").startswith("application/json") else {}


def api_register(username: str, password: str, role: str = "user", headers=None):
    r = requests.post(
        f"{API_URL}/auth/register",
        data={"username": username, "password": password, "role": role},
        headers=headers or {},
    )
    return r.status_code, r.json() if r.headers.get("content-type", "").startswith("application/json") else {}


def api_upload(file, visibility: str):
    files = {"file": (file.name, file.getvalue(), file.type)}
    data = {"visibility": visibility}
    r = requests.post(
        f"{API_URL}/upload", files=files, data=data, headers=_auth_headers()
    )
    return r.json() if r.headers.get("content-type", "").startswith("application/json") else {"error": r.text}


def api_ask(question: str):
    r = requests.post(
        f"{API_URL}/ask",
        params={"question": question},
        headers=_auth_headers(),
    )
    if r.status_code == 200:
        return r.json()
    try:
        return {"error": r.json().get("error") or r.json().get("detail") or "Unknown error"}
    except Exception:
        return {"error": r.text}


def api_summarize():
    r = requests.post(f"{API_URL}/summarize", headers=_auth_headers())
    if r.status_code == 200:
        return r.json()
    try:
        return {"error": r.json().get("error") or r.json().get("detail") or "Unknown error"}
    except Exception:
        return {"error": r.text}


def api_documents():
    r = requests.get(f"{API_URL}/documents", headers=_auth_headers())
    return r.json().get("documents", []) if r.status_code == 200 else []


def api_clear(scope: str = "mine"):
    r = requests.post(
        f"{API_URL}/clear", params={"scope": scope}, headers=_auth_headers()
    )
    return r.json() if r.status_code == 200 else {"error": r.text}


# ------------------------------------------------------------------- screens
def login_screen():
    st.set_page_config(page_title="ContextIQ", page_icon="📄", layout="centered")
    st.title("📄 ContextIQ")
    st.caption("Role-based document Q&A with retrieval-augmented generation")

    tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        st.caption(
            "Demo accounts: **admin/admin123**, **hr/hr123**, **user/user123**"
        )

        if submitted:
            status, body = api_login(username, password)
            if status == 200:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.session_state["password"] = password
                st.session_state["role"] = body.get("role", "user")
                st.rerun()
            else:
                st.error(body.get("detail") or "Login failed")

    with tab_register:
        st.write("Self-registration creates a **user** account.")
        st.caption(
            "Admins can create HR/admin accounts after logging in (sidebar → Admin)."
        )
        with st.form("register_form"):
            new_user = st.text_input("Username", key="reg_user")
            new_pass = st.text_input("Password", type="password", key="reg_pass")
            confirm = st.text_input("Confirm password", type="password", key="reg_confirm")
            reg_submitted = st.form_submit_button("Create account", use_container_width=True)

        if reg_submitted:
            if new_pass != confirm:
                st.error("Passwords don't match")
            elif len(new_pass) < 4:
                st.error("Password must be at least 4 characters")
            else:
                status, body = api_register(new_user, new_pass, role="user")
                if status == 200:
                    st.success("Account created. Log in above.")
                else:
                    st.error(body.get("detail") or "Registration failed")


def main_app():
    st.set_page_config(page_title="ContextIQ", page_icon="📄", layout="wide")
    role = st.session_state.get("role", "user")
    username = st.session_state.get("username", "?")

    role_badge = {"admin": "🛡️ Admin", "hr": "👥 HR", "user": "👤 User"}.get(
        role, role
    )
    st.title("📄 ContextIQ")
    st.caption(f"Logged in as **{username}** · {role_badge}")

    # --- Sidebar ---
    with st.sidebar:
        if st.button("🚪 Logout", use_container_width=True):
            for k in ("authenticated", "username", "password", "role"):
                st.session_state.pop(k, None)
            st.rerun()
        st.divider()

        st.header("📁 Upload")
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=SUPPORTED_TYPES,
            help="Supported: " + ", ".join(SUPPORTED_TYPES),
        )

        # Visibility options by role
        if role == "admin":
            vis_options = ["public", "team", "private"]
        elif role == "hr":
            vis_options = ["team", "private"]
        else:
            vis_options = ["private"]
        visibility = st.selectbox(
            "Visibility",
            vis_options,
            help=(
                "public = everyone can see · "
                "team = HR + admin can see · "
                "private = only you"
            ),
        )

        if uploaded_file and st.button("Process document", use_container_width=True):
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                result = api_upload(uploaded_file, visibility)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(
                    f"✅ {uploaded_file.name} processed · {result['chunks']} chunks · "
                    f"visibility: **{result['visibility']}**"
                )

        st.divider()
        st.header("📚 Visible documents")
        docs = api_documents()
        if not docs:
            st.caption("No documents visible yet.")
        else:
            for d in docs:
                if isinstance(d, dict):
                    st.write(
                        f"• **{d['filename']}** — _{d.get('visibility','?')}_ · "
                        f"by {d.get('owner_username','?')} ({d.get('owner_role','?')})"
                    )
                else:
                    st.write(f"• {d}")

        st.divider()
        st.header("🗑️ Clear")
        if st.button("Clear my documents", use_container_width=True):
            api_clear(scope="mine")
            st.success("Cleared your documents.")
            st.rerun()
        if role == "admin":
            if st.button("⚠️ Clear ALL documents", use_container_width=True):
                api_clear(scope="all")
                st.success("Cleared all documents.")
                st.rerun()

        # Admin-only: create HR/admin accounts
        if role == "admin":
            st.divider()
            with st.expander("🛡️ Admin: create account"):
                new_u = st.text_input("New username", key="admin_new_user")
                new_p = st.text_input("New password", type="password", key="admin_new_pass")
                new_r = st.selectbox("Role", ["user", "hr", "admin"], key="admin_new_role")
                if st.button("Create account", key="admin_create_btn"):
                    status, body = api_register(
                        new_u, new_p, role=new_r, headers=_auth_headers()
                    )
                    if status == 200:
                        st.success(f"Created {new_u} ({new_r})")
                    else:
                        st.error(body.get("detail") or "Create failed")

    # --- Main area ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Ask a question")
        question = st.text_input(
            "Enter your question:",
            placeholder="What does the document say about...?",
        )
        c1, c2 = st.columns(2)
        ask_clicked = c1.button("🔍 Ask", use_container_width=True)
        summarize_clicked = c2.button("📋 Summarize", use_container_width=True)

        if ask_clicked and question:
            with st.spinner("Finding answer..."):
                result = api_ask(question)
            render_answer(result)
        elif ask_clicked:
            st.warning("Please enter a question.")

        if summarize_clicked:
            with st.spinner("Generating summary..."):
                result = api_summarize()
            render_answer(result)

    with col2:
        st.subheader("💡 Tips")
        st.markdown(
            """
- Upload any supported file via the sidebar.
- **Visibility** controls who else can query it.
- Answers show a **grounding score** and inline `[source]` tags.
- Flagged `[unverified]` sentences may be hallucinations.
- Repeated or similar questions are served from the semantic cache ⚡.
            """
        )


def render_answer(result: dict):
    if "error" in result:
        st.error(result["error"])
        return
    st.subheader("📝 Answer")
    st.write(result.get("answer_with_citations") or result.get("answer", ""))

    meta_bits = []
    if result.get("cached"):
        meta_bits.append("⚡ cached answer")
    grounding = result.get("grounding")
    if grounding:
        meta_bits.append(f"🎯 grounding: {grounding['grounding_score']:.0%}")
    if result.get("path"):
        meta_bits.append(f"path: {result['path']}")
    if meta_bits:
        st.caption(" · ".join(meta_bits))

    sources = result.get("sources") or []
    if sources:
        with st.expander("📚 View sources"):
            for i, s in enumerate(sources):
                if isinstance(s, dict):
                    st.markdown(
                        f"**Source {i + 1}** — *{s.get('document','?')}* "
                        f"(score: {s.get('score', 0.0):.3f})"
                    )
                    st.info(s.get("text", ""))
                else:
                    st.info(str(s))

    if grounding and grounding.get("unsupported_count", 0) > 0:
        with st.expander("⚠️ Potentially unsupported sentences"):
            for s in grounding["sentences"]:
                if not s["supported"]:
                    st.warning(
                        f"{s['text']} _(best match score: {s['score']:.2f})_"
                    )


def main():
    if st.session_state.get("authenticated"):
        main_app()
    else:
        login_screen()


if __name__ == "__main__":
    main()
