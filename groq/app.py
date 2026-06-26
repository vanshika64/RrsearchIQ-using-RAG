import streamlit as st
import os
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="ResearchIQ",
    page_icon="📚",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* Main App */
.stApp {
    background-color: #0B0F19;
    color: white;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
    
}

/* Text Input */
.stTextInput input {
    background-color: #1F2937 !important;
    color: white !important;
    border-radius: 12px !important;
}

/* Text Area */
textarea {
    background-color: #1F2937 !important;
    color: white !important;
    border-radius: 12px !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: #6366F1;
    color: white;
    border-radius: 12px;
    height: 2.5em;
    border: none;
    font-size: 16px;
    font-weight: bold;
}

.stButton > button:hover {
    background: #818CF8;
    color: white;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background-color: #111827;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #374151;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: #111827;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- ENV ----------------

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ---------------- HEADER ----------------

st.markdown("""
<h1 style='text-align:center; color:#818CF8;'>
📚 ResearchIQ
</h1>

<h4 style='text-align:center; color:#D1D5DB;'>
AI-Powered Research Assistant
</h4>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("ResearchIQ")
    st.markdown("---")
    st.button("📚 Literature Review")
    st.button("⚖️ Compare Papers")
    st.button("🔍 Research Gaps")
    st.button("📈 Methodology Analysis")
    st.button("📝 Generate Summary")



# ---------------- CACHED LLM ----------------

@st.cache_resource
def load_llm():
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=groq_api_key
    )

# ---------------- CACHED VECTORSTORE ----------------

@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    loader = PyPDFDirectoryLoader("../papers")

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    final_documents = text_splitter.split_documents(docs)

    vectors = FAISS.from_documents(
        final_documents,
        embeddings
    )

    return vectors

# ---------------- LOAD RESOURCES ----------------

llm = load_llm()

with st.spinner("Loading Research Papers..."):
    vectors = load_vectorstore()

# ---------------- PROMPT ----------------

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context.

    If the answer exists in the research papers,
    answer using the context.

    If the answer is not found in the documents,
    clearly state that the information is not present.And after that explain acc to ur knowledge.

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# ---------------- INPUT ----------------

question = st.text_area(
    "Ask anything about your research papers",
    height=120,
    placeholder="Example: What methodology is proposed in Paper 3?"
)

# ---------------- QUERY ----------------

if st.button("🚀 Get Answer"):

    if question.strip() == "":
        st.warning("Please enter a question.")
        st.stop()

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retriever = vectors.as_retriever()

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    with st.spinner("Analyzing Research Papers..."):

        start = time.process_time()

        response = retrieval_chain.invoke(
            {"input": question}
        )

        end = time.process_time()

    # ---------------- ANSWER CARD ----------------

    st.markdown(
        f"""
        <div style="
        background:#111827;
        padding:20px;
        border-radius:15px;
        border:1px solid #4F46E5;
        margin-top:15px;
        ">
        <h3 style="color:#818CF8;">Answer </h3>
        <p style="color:white;font-size:17px;">{response['answer']}</p></div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        f"⏱ Response Time: {end - start:.2f} sec"
    )

    # ---------------- SOURCES ----------------

    with st.expander("📄 Document Similarity Search"):

        for i, doc in enumerate(response["context"]):

            st.markdown(
                f"### Source {i+1}"
            )

            st.write(doc.page_content)

            st.markdown("---")

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
    """
    <center>
    Built with using LangChain, FAISS, Groq & Streamlit
    </center>
    """,
    unsafe_allow_html=True
)