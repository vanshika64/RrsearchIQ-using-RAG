import sys
from pathlib import Path

import streamlit as st

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import rag_service

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

    div[data-baseweb="textarea"] {
        border: 2px solid #374151 !important;
        border-radius: 12px !important;
        background: #1F2937 !important;
    }

    /* Focus state */
    div[data-baseweb="textarea"]:focus-within {
        border: 2px solid #6366F1 !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.25) !important;
    }

    /* Actual textarea */
    div[data-baseweb="textarea"] textarea {
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="brand-title">Research IQ</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="brand-subtitle">AI-Powered Research Assistant</p>',
    unsafe_allow_html=True,
)

papers = rag_service.list_papers()

with st.sidebar:
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
                rag_service.save_paper(uploaded.name, uploaded.getbuffer())
                saved += 1
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
                st.caption(f"{paper['size_kb']} KB")
            with col2:
                if st.button("Remove", key=f"remove_{paper['filename']}"):
                    rag_service.delete_paper(paper["filename"])
                    st.rerun()


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
                result = rag_service.query_papers(question.strip())
            except ValueError as exc:
                st.error(str(exc))
                st.stop()
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()
            except Exception:
                st.error("Failed to process your question. Please try again.")
                st.stop()

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

st.markdown("---")
st.markdown(
    "<center>Built with Streamlit, FastAPI, LangChain, FAISS & Groq</center>",
    unsafe_allow_html=True,
)
