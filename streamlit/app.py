import os
from pathlib import Path

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Research IQ",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    .stApp {
        background-color: #0B0F19;
        color: white;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    .brand-title {
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: 6rem;
        font-weight: 700;
        font-size: 100px !important;
        text-align: center;
        margin: 0;
        background: linear-gradient(135deg, #c7d2fe 0%, #818cf8 45%, #6366f1 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .brand-subtitle {
        text-align: center;
        color: #9CA3AF;
        font-family: 'Inter', sans-serif;
        margin-top: 0.5rem;
    }

    div[data-baseweb="textarea"], div[data-baseweb="input"] {
        border: 2px solid #374151 !important;
        border-radius: 12px !important;
        background: #1F2937 !important;
    }

    div[data-baseweb="textarea"]:focus-within, div[data-baseweb="input"]:focus-within {
        border: 2px solid #6366F1 !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.25) !important;
    }

    div[data-baseweb="textarea"] textarea, div[data-baseweb="input"] input {
        background: transparent !important;
        color: white !important;
    }

    .stButton > button {
        background: #6366F1;
        color: white;
        border-radius: 12px;
        border: none;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #818CF8;
        color: white;
    }

    .answer-box {
        background: #111827;
        padding: 1.25rem;
        border-radius: 15px;
        border: 1px solid #4F46E5;
        margin-top: 1rem;
    }

    .answer-box h3 {
        color: #818CF8;
        margin-top: 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1F2937;
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1.25rem;
        color: #9CA3AF;
    }

    .stTabs [aria-selected="true"] {
        background-color: #111827;
        color: #818CF8 !important;
        border-bottom: 2px solid #6366F1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="brand-title">Research IQ</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="brand-subtitle">AI-Powered Research Assistant</p>',
    unsafe_allow_html=True,
)

if "token" not in st.session_state:
    st.session_state.token = None
    st.session_state.user_name = None


def _error_detail(response: requests.Response, fallback: str) -> str:
    try:
        return response.json().get("detail", fallback)
    except ValueError:
        return fallback


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


# --------------------------------------------------------------------------
# Logged-out view: Log In / Sign Up
# --------------------------------------------------------------------------
if not st.session_state.token:
    login_tab, signup_tab = st.tabs(["Log In", "Sign Up"])

    with login_tab:
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Log In", use_container_width=True):
            if not login_email or not login_password:
                st.warning("Enter your email and password.")
            else:
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/auth/login",
                        data={"username": login_email, "password": login_password},
                        timeout=30,
                    )
                except requests.RequestException:
                    st.error("Could not reach the backend. Is it running?")
                else:
                    if response.status_code == 200:
                        st.session_state.token = response.json()["access_token"]
                        st.session_state.user_name = login_email
                        st.rerun()
                    else:
                        st.error(_error_detail(response, "Login failed."))

    with signup_tab:
        signup_name = st.text_input("Name", key="signup_name")
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password", type="password", key="signup_password")

        if st.button("Create Account", use_container_width=True):
            if not (signup_name and signup_email and signup_password):
                st.warning("Fill in all fields.")
            elif len(signup_password) < 8:
                st.warning("Password must be at least 8 characters.")
            else:
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/auth/signup",
                        json={
                            "name": signup_name,
                            "email": signup_email,
                            "password": signup_password,
                        },
                        timeout=30,
                    )
                except requests.RequestException:
                    st.error("Could not reach the backend. Is it running?")
                else:
                    if response.status_code == 201:
                        st.success("Account created — you can log in now.")
                    else:
                        st.error(_error_detail(response, "Sign up failed."))

    st.stop()

# --------------------------------------------------------------------------
# Logged-in view
# --------------------------------------------------------------------------


def fetch_papers() -> list[dict]:
    try:
        response = requests.get(f"{API_BASE_URL}/papers", headers=auth_headers(), timeout=30)
    except requests.RequestException:
        st.error("Could not reach the backend.")
        return []

    if response.status_code == 200:
        return response.json()
    if response.status_code == 401:
        st.session_state.token = None
        st.rerun()
    st.error(_error_detail(response, "Failed to load papers."))
    return []


papers = fetch_papers()

with st.sidebar:
    st.write(f"Logged in as **{st.session_state.user_name}**")
    if st.button("Log Out", use_container_width=True):
        st.session_state.token = None
        st.session_state.user_name = None
        st.rerun()

    st.markdown("---")
    st.header("Upload")
    uploaded_files = st.file_uploader(
        "Upload Research Paper",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("Save uploaded papers", use_container_width=True):
        if not uploaded_files:
            st.warning("Choose at least one PDF to upload.")
        else:
            saved = 0
            for uploaded in uploaded_files:
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/papers",
                        headers=auth_headers(),
                        files=files,
                        timeout=120,
                    )
                except requests.RequestException:
                    st.error("Could not reach the backend.")
                    break

                if response.status_code == 201:
                    saved += 1
                else:
                    st.error(f"{uploaded.name}: {_error_detail(response, 'Upload failed.')}")

            if saved:
                st.success(f"Saved {saved} paper(s).")
                st.rerun()

    st.markdown("---")
    st.subheader("Your Papers")
    if not papers:
        st.caption("No papers uploaded yet.")
    else:
        for paper in papers:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(paper["filename"])
            with col2:
                if st.button("Remove", key=f"remove_{paper['filename']}"):
                    try:
                        response = requests.delete(
                            f"{API_BASE_URL}/papers/{paper['filename']}",
                            headers=auth_headers(),
                            timeout=30,
                        )
                    except requests.RequestException:
                        st.error("Could not reach the backend.")
                    else:
                        if response.status_code == 200:
                            st.rerun()
                        else:
                            st.error(_error_detail(response, "Failed to remove paper."))


ask_tab, summarize_tab = st.tabs(["Ask", "Summarize"])

with ask_tab:
    question = st.text_area(
        "Ask anything about your research papers",
        height=140,
        placeholder="Example: What methodology is proposed in the paper?",
    )

    if st.button("Get Answer", use_container_width=True):
        if not papers:
            st.warning("Upload at least one research paper first.")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing research papers..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/query",
                        headers=auth_headers(),
                        json={"question": question.strip()},
                        timeout=120,
                    )
                except requests.RequestException:
                    st.error("Could not reach the backend.")
                    st.stop()

            if response.status_code != 200:
                st.error(_error_detail(response, "Failed to process your question."))
                st.stop()

            result = response.json()
            st.markdown("### Answer")
            st.markdown(result["answer"])
            st.caption(f"Response time: {result['response_time_sec']:.2f} sec")

            if result["sources"]:
                with st.expander("Sources"):
                    for i, source in enumerate(result["sources"], start=1):
                        source_name = source.get("metadata", {}).get("source", "")
                        if source_name:
                            source_name = Path(source_name).name
                        st.markdown(f"**Source {i}** {f'— {source_name}' if source_name else ''}")
                        st.write(source["content"])
                        st.markdown("---")

with summarize_tab:
    if not papers:
        st.warning("Upload at least one research paper first.")
    else:
        filenames = [paper["filename"] for paper in papers]
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_paper = st.selectbox("Choose a paper to summarize", filenames)
        with col2:
            summary_length = st.selectbox("Length", ["brief", "detailed"])

        if st.button("Generate Summary", use_container_width=True):
            with st.spinner(f"Summarizing {selected_paper}..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/summarize",
                        headers=auth_headers(),
                        json={"filename": selected_paper, "length": summary_length},
                        timeout=180,
                    )
                except requests.RequestException:
                    st.error("Could not reach the backend.")
                    st.stop()

            if response.status_code != 200:
                st.error(_error_detail(response, "Failed to summarize this paper."))
                st.stop()

            result = response.json()
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.markdown(f"### Summary — {selected_paper}")
            st.markdown(result["summary"])
            st.markdown("</div>", unsafe_allow_html=True)
            st.caption(
                f"Response time: {result['response_time_sec']:.2f} sec · "
                f"{result['chunks_processed']} chunk(s) processed"
                + (" via map-reduce" if result["used_map_reduce"] else "")
            )

st.markdown("---")
st.markdown(
    "<center>Built with Streamlit, FastAPI, LangChain, FAISS, Postgres & S3</center>",
    unsafe_allow_html=True,
)