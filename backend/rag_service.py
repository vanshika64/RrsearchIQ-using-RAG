import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from . import models
from . import storage
from . import vectorstore
from .config import CHUNK_OVERLAP, CHUNK_SIZE, GROQ_API_KEY, GROQ_MODEL, TMP_DIR

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

SUMMARY_MAP_PROMPT = ChatPromptTemplate.from_template(
    """
    Summarize the key points, findings, and methodology described in this
    excerpt from a research paper. Be concise and factual. Do not add
    information that isn't in the excerpt.

    <excerpt>
    {text}
    </excerpt>

    Summary:
    """
)

SUMMARY_REDUCE_PROMPT = ChatPromptTemplate.from_template(
    """
    You are writing a summary of a research paper based on either its full
    text or a set of partial section summaries below.

    Write a {length_instruction} summary that covers, where present in the
    source material:
    - The paper's main objective / research question
    - Methodology
    - Key findings and results
    - Conclusions and implications

    <source_material>
    {text}
    </source_material>

    Final summary:
    """
)

SUMMARY_LENGTH_INSTRUCTIONS = {
    "brief": "short, 3-5 sentence",
    "detailed": "detailed, multi-paragraph",
}

# Papers with more chunks than this go through a map-reduce summarization
# pass instead of a single direct call, to stay within context limits.
MAP_REDUCE_CHUNK_THRESHOLD = 4


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in the environment.")
    return ChatGroq(model_name=GROQ_MODEL, api_key=GROQ_API_KEY)


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def _load_pdf_bytes_as_documents(filename: str, content: bytes):
    """PyPDFLoader needs a real file path, so the S3 bytes are staged to a
    scratch file first and removed again immediately after loading."""
    scratch_path = TMP_DIR / f"{uuid.uuid4().hex}_{filename}"
    scratch_path.write_bytes(content)
    try:
        docs = PyPDFLoader(str(scratch_path)).load()
    finally:
        scratch_path.unlink(missing_ok=True)

    for doc in docs:
        doc.metadata["source"] = filename
    return docs


def list_papers(db: Session, user_id: uuid.UUID) -> list[models.Paper]:
    return (
        db.query(models.Paper)
        .filter(models.Paper.user_id == user_id)
        .order_by(models.Paper.filename)
        .all()
    )


def save_paper(
    db: Session, user_id: uuid.UUID, user_key: str, filename: str, content: bytes
) -> models.Paper:
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    existing = (
        db.query(models.Paper)
        .filter(models.Paper.user_id == user_id, models.Paper.filename == safe_name)
        .first()
    )
    if existing:
        raise ValueError(f"You've already uploaded a paper named '{safe_name}'.")

    storage_url = storage.upload_paper(user_key, safe_name, content)

    paper = models.Paper(user_id=user_id, filename=safe_name, storage_url=storage_url)
    db.add(paper)
    db.commit()
    db.refresh(paper)

    try:
        docs = _load_pdf_bytes_as_documents(safe_name, content)
        chunks = _splitter().split_documents(docs)
        vectorstore.add_documents(user_key, chunks)
    except Exception:
        # Keep the DB/S3 and vector index consistent if embedding fails.
        db.delete(paper)
        db.commit()
        storage.delete_paper_object(user_key, safe_name)
        raise

    return paper


def delete_paper(db: Session, user_id: uuid.UUID, user_key: str, filename: str) -> bool:
    safe_name = Path(filename).name
    paper = (
        db.query(models.Paper)
        .filter(models.Paper.user_id == user_id, models.Paper.filename == safe_name)
        .first()
    )
    if not paper:
        return False

    storage.delete_paper_object(user_key, safe_name)
    db.delete(paper)
    db.commit()

    _rebuild_user_index(db, user_id, user_key)
    return True


def _rebuild_user_index(db: Session, user_id: uuid.UUID, user_key: str) -> None:
    remaining = db.query(models.Paper).filter(models.Paper.user_id == user_id).all()

    all_chunks = []
    for paper in remaining:
        content = storage.download_paper(user_key, paper.filename)
        docs = _load_pdf_bytes_as_documents(paper.filename, content)
        all_chunks.extend(_splitter().split_documents(docs))

    vectorstore.rebuild_index(user_key, all_chunks)


def query_papers(user_key: str, question: str) -> dict[str, Any]:
    vectors = vectorstore.load_index(user_key)
    if vectors is None:
        raise ValueError("Upload at least one research paper before asking questions.")

    llm = get_llm()
    retriever = vectors.as_retriever()

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    start = time.process_time()
    source_docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)
    elapsed = time.process_time() - start

    sources = [{"content": d.page_content, "metadata": d.metadata} for d in source_docs]

    return {"answer": answer, "response_time_sec": round(elapsed, 2), "sources": sources}


def summarize_paper(user_key: str, filename: str, length: str = "brief") -> dict[str, Any]:
    safe_name = Path(filename).name
    content = storage.download_paper(user_key, safe_name)
    docs = _load_pdf_bytes_as_documents(safe_name, content)
    if not docs:
        raise ValueError("Could not extract any text from this paper.")

    chunks = _splitter().split_documents(docs)

    llm = get_llm()
    length_instruction = SUMMARY_LENGTH_INSTRUCTIONS.get(
        length, SUMMARY_LENGTH_INSTRUCTIONS["brief"]
    )
    reduce_chain = SUMMARY_REDUCE_PROMPT | llm | StrOutputParser()

    start = time.process_time()

    if len(chunks) <= MAP_REDUCE_CHUNK_THRESHOLD:
        source_text = format_docs(chunks)
        used_map_reduce = False
    else:
        map_chain = SUMMARY_MAP_PROMPT | llm | StrOutputParser()
        partial_summaries = [
            map_chain.invoke({"text": chunk.page_content}) for chunk in chunks
        ]
        source_text = "\n\n".join(partial_summaries)
        used_map_reduce = True

    summary = reduce_chain.invoke(
        {"text": source_text, "length_instruction": length_instruction}
    )
    elapsed = time.process_time() - start

    return {
        "summary": summary,
        "chunks_processed": len(chunks),
        "used_map_reduce": used_map_reduce,
        "response_time_sec": round(elapsed, 2),
    }