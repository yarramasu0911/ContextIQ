import json
import os

import requests
import streamlit as st

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
    tok = st.session_state.get("token")
    if tok:
        return {"Authorization": f"Bearer {tok}"}
    # Fallback to basic headers (legacy)
    if st.session_state.get("authenticated"):
        return {
            "X-Username": st.session_state["username"],
            "X-Password": st.session_state.get("password", ""),
        }
    return {}


def _safe_json(r):
    try:
        return r.json()
    except Exception:
        return {"error": r.text}


def api_login(username, password):
    r = requests.post(
        f"{API_URL}/auth/login",
        data={"username": username, "password": password},
    )
    return r.status_code, _safe_json(r)


def api_register(username, password, role="user", headers=None):
    r = requests.post(
        f"{API_URL}/auth/register",
        data={"username": username, "password": password, "role": role},
        headers=headers or {},
    )
    return r.status_code, _safe_json(r)


def api_upload(file, visibility):
    files = {"file": (file.name, file.getvalue(), file.type)}
    data = {"visibility": visibility}
    r = requests.post(
        f"{API_URL}/upload", files=files, data=data, headers=_auth_headers()
    )
    return _safe_json(r)


def api_ask(question):
    r = requests.post(
        f"{API_URL}/ask", params={"question": question}, headers=_auth_headers()
    )
    if r.status_code == 200:
        return _safe_json(r)
    body = _safe_json(r)
    return {"error": body.get("error") or body.get("detail") or "Unknown error"}


def api_ask_stream(question):
    """Generator yielding (type, payload) tuples: meta, token, done."""
    with requests.post(
        f"{API_URL}/ask/stream",
        params={"question": question},
        headers=_auth_headers(),
        stream=True,
    ) as r:
        if r.status_code != 200:
            body = _safe_json(r)
            yield "error", body.get("detail") or body.get("error") or r.text
            return
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            try:
                evt = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            yield evt.get("type"), evt.get("payload")


def api_summarize():
    r = requests.post(f"{API_URL}/summarize", headers=_auth_headers())
    if r.status_code == 200:
        return _safe_json(r)
    body = _safe_json(r)
    return {"error": body.get("error") or body.get("detail") or "Unknown error"}


def api_documents():
    r = requests.get(f"{API_URL}/documents", headers=_auth_headers())
    return _safe_json(r).get("documents", []) if r.status_code == 200 else []


def api_clear(scope="mine"):
    requests.post(
        f"{API_URL}/clear", params={"scope": scope}, headers=_auth_headers()
    )


def api_list_users():
    r = requests.get(f"{API_URL}/admin/users", headers=_auth_headers())
    return _safe_json(r).get("users", []) if r.status_code == 200 else []


def api_change_role(username, role):
    r = requests.post(
        f"{API_URL}/admin/users/role",
        data={"username": username, "role": role},
        headers=_auth_headers(),
    )
    return _safe_json(r)


def api_deactivate(username):
    r = requests.post(
        f"{API_URL}/admin/users/deactivate",
        data={"username": username},
        headers=_auth_headers(),
    )
    return _safe_json(r)


def api_audit(n=50):
    r = requests.get(
        f"{API_URL}/admin/audit", params={"n": n}, headers=_auth_headers()
    )
    return _safe_json(r).get("entries", []) if r.status_code == 200 else []


def api_traces(n=20):
    r = requests.get(
        f"{API_URL}/debug/trace", params={"n": n}, headers=_auth_headers()
    )
    return _safe_json(r).get("traces", []) if r.status_code == 200 else []


# ------------------------------------------------------------------ screens
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
                st.session_state["role"] = body.get("role", "user")
                st.session_state["token"] = body.get("token")
                st.rerun()
            else:
                st.error(body.get("detail") or "Login failed")

    with tab_register:
        st.write("Self-registration creates a **user** account.")
        st.caption("Admins can create HR/admin accounts after logging in.")
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


# ------------------------------------------------------------------ rendering
def render_answer(result):
    if "error" in result:
        st.error(result["error"])
        return
    st.write(result.get("answer_with_citations") or result.get("answer", ""))

    meta_bits = []
    if result.get("cached"):
        meta_bits.append("⚡ cached")
    if result.get("intent") and result["intent"] != "other":
        meta_bits.append(f"🎯 {result['intent']}")
    grounding = result.get("grounding")
    if grounding:
        meta_bits.append(f"faithfulness: {grounding['grounding_score']:.0%}")
    if result.get("path"):
        meta_bits.append(f"path: {result['path']}")
    if meta_bits:
        st.caption(" · ".join(meta_bits))

    sources = result.get("sources") or []
    if sources:
        with st.expander("📚 Sources"):
            for i, s in enumerate(sources):
                if isinstance(s, dict):
                    st.markdown(
                        f"**Source {i+1}** — *{s.get('document','?')}* "
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
                        f"{s['text']} _(best match: {s['score']:.2f})_"
                    )


# ----------------------------------------------------------- main application
def sidebar(role, username):
    with st.sidebar:
        if st.button("🚪 Logout", use_container_width=True):
            for k in ("authenticated", "username", "password", "role", "token", "chat_history"):
                st.session_state.pop(k, None)
            st.rerun()
        st.divider()

        st.header("📁 Upload")
        uploaded = st.file_uploader(
            "Upload a document",
            type=SUPPORTED_TYPES,
            help="Supported: " + ", ".join(SUPPORTED_TYPES),
        )
        if role == "admin":
            vis_opts = ["public", "team", "private"]
        elif role == "hr":
            vis_opts = ["team", "private"]
        else:
            vis_opts = ["private"]
        vis = st.selectbox(
            "Visibility", vis_opts,
            help="public=everyone · team=HR+admin · private=only you",
        )
        if uploaded and st.button("Process", use_container_width=True):
            with st.spinner(f"Processing {uploaded.name}..."):
                r = api_upload(uploaded, vis)
            if "error" in r:
                st.error(r["error"])
            else:
                st.success(
                    f"✅ {uploaded.name} · {r['chunks']} chunks · {r['visibility']}"
                )

        st.divider()
        st.header("📚 Visible documents")
        docs = api_documents()
        if not docs:
            st.caption("Nothing uploaded yet.")
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
        if st.button("🗑️ Clear my documents", use_container_width=True):
            api_clear("mine")
            st.rerun()
        if role == "admin":
            if st.button("⚠️ Clear ALL documents", use_container_width=True):
                api_clear("all")
                st.rerun()

        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()


def admin_panel():
    st.subheader("🛡️ Admin panel")

    with st.expander("👥 Users", expanded=True):
        users = api_list_users()
        if not users:
            st.caption("No users.")
        for u in users:
            cols = st.columns([2, 2, 2, 1])
            cols[0].write(f"**{u['username']}**")
            cols[1].write(u["role"])
            cols[2].write("🟢 active" if u.get("active", True) else "🔴 disabled")
            if cols[3].button("edit", key=f"edit_{u['username']}"):
                st.session_state["edit_user"] = u["username"]

        target = st.session_state.get("edit_user")
        if target:
            st.markdown(f"**Editing:** {target}")
            new_role = st.selectbox("New role", ["user", "hr", "admin"], key="new_role")
            c1, c2, c3 = st.columns(3)
            if c1.button("Change role"):
                body = api_change_role(target, new_role)
                st.success(body.get("message") or "updated")
                st.rerun()
            if c2.button("Deactivate"):
                body = api_deactivate(target)
                st.warning(body.get("message") or "deactivated")
                st.rerun()
            if c3.button("Cancel"):
                st.session_state.pop("edit_user", None)
                st.rerun()

    with st.expander("📋 Create account"):
        new_u = st.text_input("Username", key="admin_new_u")
        new_p = st.text_input("Password", type="password", key="admin_new_p")
        new_r = st.selectbox("Role", ["user", "hr", "admin"], key="admin_new_r")
        if st.button("Create"):
            status, body = api_register(new_u, new_p, role=new_r, headers=_auth_headers())
            if status == 200:
                st.success(f"Created {new_u} ({new_r})")
            else:
                st.error(body.get("detail") or "Failed")

    with st.expander("🔍 Recent traces (admin-only)"):
        traces = api_traces(20)
        if not traces:
            st.caption("No traces yet.")
        for t in traces:
            st.markdown(
                f"- **{t.get('intent','?')}** `{t.get('question','')[:80]}` "
                f"by {t.get('viewer_username')} ({t.get('viewer_role')}) · "
                f"grounding: {t.get('grounding_score')}"
            )


def audit_panel():
    st.subheader("📋 Audit log")
    rows = api_audit(100)
    if not rows:
        st.caption("No audit entries yet.")
        return
    for r in reversed(rows):
        st.markdown(
            f"- **{r.get('username')}** ({r.get('role')}) · "
            f"*{r.get('intent','?')}* · "
            f"{r.get('question','')[:100]} "
            f"→ docs: {', '.join(r.get('documents') or [])} · "
            f"grounding: {r.get('grounding_score')}"
        )


def chat_tab(role, username, stream_enabled):
    history = st.session_state.setdefault("chat_history", [])
    for msg in history:
        with st.chat_message(msg["role"]):
            st.write(msg.get("content_html") or msg.get("content", ""))
            if msg.get("result"):
                render_answer(msg["result"])

    if question := st.chat_input("Ask about your documents..."):
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            if stream_enabled:
                placeholder = st.empty()
                buffer = ""
                meta_result = None
                for ev_type, payload in api_ask_stream(question):
                    if ev_type == "meta":
                        meta_result = payload
                    elif ev_type == "token":
                        buffer += payload
                        placeholder.markdown(buffer)
                    elif ev_type == "error":
                        st.error(payload)
                        return
                # finalize
                final = dict(meta_result or {})
                final["answer"] = buffer
                history.append({"role": "assistant", "content": buffer, "result": final})
                render_answer(final)
            else:
                with st.spinner("Thinking..."):
                    result = api_ask(question)
                history.append(
                    {
                        "role": "assistant",
                        "content": result.get("answer_with_citations") or result.get("answer", ""),
                        "result": result,
                    }
                )
                render_answer(result)


def main_app():
    st.set_page_config(page_title="ContextIQ", page_icon="📄", layout="wide")
    role = st.session_state.get("role", "user")
    username = st.session_state.get("username", "?")

    badge = {"admin": "🛡️ Admin", "hr": "👥 HR", "user": "👤 User"}.get(role, role)
    st.title("📄 ContextIQ")
    st.caption(f"Logged in as **{username}** · {badge}")

    sidebar(role, username)

    # Admin/HR get extra tabs
    tabs = ["💬 Chat"]
    if role in ("admin", "hr"):
        tabs.append("📋 Audit")
    if role == "admin":
        tabs.append("🛡️ Admin")

    stream_toggle = st.toggle("Stream answers", value=False)
    tab_objs = st.tabs(tabs)

    with tab_objs[0]:
        chat_tab(role, username, stream_toggle)

    idx = 1
    if role in ("admin", "hr"):
        with tab_objs[idx]:
            audit_panel()
        idx += 1
    if role == "admin":
        with tab_objs[idx]:
            admin_panel()


def main():
    if st.session_state.get("authenticated"):
        main_app()
    else:
        login_screen()


if __name__ == "__main__":
    main()
