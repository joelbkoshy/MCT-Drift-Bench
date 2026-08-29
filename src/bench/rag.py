"""Retrieval over the MCT manual.

Identical to the retrieval component used in the earlier phases: the manual is
chunked, embedded with all-MiniLM-L6-v2, stored in Chroma, and the top-k
passages for a query are concatenated into the prompt. Held constant here so
that the retrieval arms differ from the non-retrieval arms in one thing only.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parents[2]
MANUAL_PATH = BASE_DIR / "data" / "mct_manual.txt"
CHROMA_DIR = BASE_DIR / "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3


@lru_cache(maxsize=1)
def _store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
    )
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)

    text = MANUAL_PATH.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n### ", "\n## ", "\n\n", "\n", ". ", " "],
    )
    return Chroma.from_texts(
        texts=splitter.split_text(text),
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def retrieve(query: str, top_k: int = TOP_K) -> str:
    docs = _store().similarity_search(query, k=top_k)
    return "\n\n---\n\n".join(d.page_content for d in docs)


def chunk_count() -> int:
    return _store()._collection.count()
