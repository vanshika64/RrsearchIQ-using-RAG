from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from .config import EMBEDDING_MODEL, VECTOR_STORE_DIR


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _user_index_dir(user_key: str) -> Path:
    path = VECTOR_STORE_DIR / user_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def index_exists(user_key: str) -> bool:
    return (_user_index_dir(user_key) / "index.faiss").exists()


def load_index(user_key: str) -> FAISS | None:
    if not index_exists(user_key):
        return None
    return FAISS.load_local(
        str(_user_index_dir(user_key)),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def add_documents(user_key: str, documents: list) -> FAISS:
    """Adds documents to (or creates) this user's FAISS index and persists it."""
    index_dir = _user_index_dir(user_key)
    existing = load_index(user_key)

    if existing is None:
        vectors = FAISS.from_documents(documents, get_embeddings())
    else:
        existing.add_documents(documents)
        vectors = existing

    vectors.save_local(str(index_dir))
    return vectors


def rebuild_index(user_key: str, documents: list) -> FAISS | None:
    """Rebuilds this user's index from scratch (used after a paper is deleted)."""
    index_dir = _user_index_dir(user_key)
    for item in index_dir.glob("*"):
        item.unlink()

    if not documents:
        return None

    vectors = FAISS.from_documents(documents, get_embeddings())
    vectors.save_local(str(index_dir))
    return vectors