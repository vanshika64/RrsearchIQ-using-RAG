import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    FAISS_DIR,
    GROQ_API_KEY,
    GROQ_MODEL,
    PAPERS_DIR,
)

PROMPT = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context.

    If the answer exists in the research papers, answer using the context.
    If the answer is not found in the documents, clearly state that the
    information is not present. After that, explain according to your knowledge.

    <context>
    {context}
    </context>

    Question: {input}
    """
)


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in the environment.")
    return ChatGroq(model_name=GROQ_MODEL, api_key=GROQ_API_KEY)


def list_papers() -> list[dict[str, Any]]:
    papers = []
    for pdf in sorted(PAPERS_DIR.glob("*.pdf")):
        stat = pdf.stat()
        papers.append(
            {
                "filename": pdf.name,
                "size_kb": round(stat.st_size / 1024, 1),
            }
        )
    return papers


def has_papers() -> bool:
    return any(PAPERS_DIR.glob("*.pdf"))


def save_paper(filename: str, content: bytes) -> str:
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    destination = PAPERS_DIR / safe_name
    destination.write_bytes(content)
    rebuild_index()
    return safe_name


def delete_paper(filename: str) -> bool:
    target = PAPERS_DIR / Path(filename).name
    if not target.exists():
        return False
    target.unlink()
    rebuild_index()
    return True


def rebuild_index() -> None:
    get_vectorstore.cache_clear()
    if not has_papers():
        index_file = FAISS_DIR / "index.faiss"
        if index_file.exists():
            for item in FAISS_DIR.iterdir():
                item.unlink()
        return
    get_vectorstore()


@lru_cache(maxsize=1)
def get_vectorstore() -> FAISS | None:
    if not has_papers():
        return None

    loader = PyPDFDirectoryLoader(str(PAPERS_DIR))
    docs = loader.load()

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    vectors = FAISS.from_documents(chunks, get_embeddings())
    vectors.save_local(str(FAISS_DIR))
    return vectors


def query_papers(question: str) -> dict[str, Any]:
    vectors = get_vectorstore()
    if vectors is None:
        raise ValueError("Upload at least one research paper before asking questions.")

    llm = get_llm()
    retriever = vectors.as_retriever()

    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    start = time.process_time()
    source_docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)
    elapsed = time.process_time() - start

    sources = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in source_docs
    ]

    return {
        "answer": answer,
        "response_time_sec": round(elapsed, 2),
        "sources": sources,
    }
