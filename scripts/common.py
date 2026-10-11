from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma_store"
BM25_PATH = DATA_DIR / "bm25_index.pkl"
REGISTRY_PATH = DATA_DIR / "multi_doc_registry.json"
COLLECTION_NAME = "pdf_chunks"
EMBEDDING_MODEL = os.getenv("DOCUMENT_READER_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
RERANKER_MODEL = os.getenv("DOCUMENT_READER_RERANKER_MODEL", "BAAI/bge-reranker-base")
LLM_MODEL = os.getenv("DOCUMENT_READER_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_registry() -> dict[str, Any]:
    ensure_dirs()
    if not REGISTRY_PATH.exists():
        return {"schema_version": 1, "embedding_model": EMBEDDING_MODEL, "documents": {}}
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        registry = json.load(f)
    if registry.get("embedding_model") != EMBEDDING_MODEL and registry.get("documents"):
        raise RuntimeError(
            "The configured embedding model differs from the indexed model. "
            "Re-index every PDF with --force or restore DOCUMENT_READER_EMBEDDING_MODEL."
        )
    return registry


def save_registry(registry: dict[str, Any]) -> None:
    ensure_dirs()
    tmp = REGISTRY_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=True)
    tmp.replace(REGISTRY_PATH)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_doc_id(path: Path) -> str:
    # Stable across runs; different files with the same basename remain distinct.
    return hashlib.sha1(str(path.resolve()).lower().encode("utf-8")).hexdigest()[:16]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_:.#/-]+", text.lower())


def is_identifier_query(query: str) -> bool:
    return bool(re.search(r"\b[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+\b", query)
                or re.search(r"\b[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+\b", query)
                or re.search(r"\b[A-Za-z_][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b", query))
