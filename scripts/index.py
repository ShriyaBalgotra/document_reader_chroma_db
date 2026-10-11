from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import fitz
import chromadb
from rank_bm25 import BM25Okapi

from common import (
    BM25_PATH, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, ROOT,
    ensure_dirs, load_registry, make_doc_id, save_registry, sha256_file, tokenize
)

CHUNK_WORDS = 300
OVERLAP_WORDS = 50


def split_page_text(text: str, page_number: int) -> list[dict]:
    words = text.split()
    if not words:
        return []
    step = CHUNK_WORDS - OVERLAP_WORDS
    chunks = []
    for start in range(0, len(words), step):
        part = words[start:start + CHUNK_WORDS]
        if not part:
            break
        chunks.append({
            "text": " ".join(part),
            "start_page": page_number,
            "end_page": page_number,
        })
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def extract_chunks(pdf_path: Path) -> list[dict]:
    chunks = []
    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            for chunk in split_page_text(text, page_index):
                chunks.append(chunk)
    return chunks


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def rebuild_bm25(collection) -> int:
    all_rows = collection.get(include=["documents", "metadatas"])
    docs = all_rows.get("documents") or []
    ids = all_rows.get("ids") or []
    metadatas = all_rows.get("metadatas") or []
    corpus = [tokenize(text or "") for text in docs]
    bm25 = BM25Okapi(corpus) if corpus else None
    ensure_dirs()
    with BM25_PATH.open("wb") as f:
        pickle.dump({
            "ids": ids,
            "documents": docs,
            "metadatas": metadatas,
            "tokenized_corpus": corpus,
            "bm25": bm25,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    return len(ids)


def status(path: str | None = None) -> None:
    registry = load_registry()
    docs = registry.get("documents", {})
    if path:
        target = Path(path).resolve()
        doc_id = make_doc_id(target)
        entry = docs.get(doc_id)
        if entry:
            print(f"Indexed: {entry['source_path']}")
            print(f"Chunks: {entry['chunk_count']}")
            print(f"Indexed at: {entry['indexed_at']}")
        else:
            print("This PDF is not indexed.")
        return
    if not docs:
        print("No PDFs indexed yet.")
        return
    for doc_id, entry in docs.items():
        print(f"- {entry['source_path']} | chunks={entry['chunk_count']} | doc_id={doc_id}")


def index_pdf(input_path: str, force: bool = False) -> None:
    pdf_path = Path(input_path).expanduser().resolve()
    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")
    registry = load_registry()
    doc_id = make_doc_id(pdf_path)
    fingerprint = sha256_file(pdf_path)
    old_entry = registry.get("documents", {}).get(doc_id)
    if old_entry and old_entry.get("sha256") == fingerprint and not force:
        print("This exact PDF is already indexed. Use --force to rebuild it.")
        return
    if old_entry and not force:
        raise RuntimeError("This PDF changed since indexing. Use --force to replace its chunks.")

    chunks = extract_chunks(pdf_path)
    if not chunks:
        raise RuntimeError(
            "No extractable text found. This may be a scanned PDF; OCR is not included in this version."
        )

    ensure_dirs()
    collection = get_collection()
    # Replace only this document's chunks; other indexed PDFs remain intact.
    if old_entry:
        collection.delete(where={"doc_id": doc_id})

    ids, documents, metadatas = [], [], []
    for i, chunk in enumerate(chunks):
        ids.append(f"{doc_id}:{i:06d}")
        documents.append(chunk["text"])
        metadatas.append({
            "doc_id": doc_id,
            "source_name": pdf_path.name,
            "source_path": str(pdf_path),
            "chunk_index": i,
            "start_page": chunk["start_page"],
            "end_page": chunk["end_page"],
            "embedding_model": EMBEDDING_MODEL,
        })
    # Chroma's embedding function is explicit to ensure query/index consistency.
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=16,
    ).tolist()
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    from datetime import datetime, timezone
    registry.setdefault("documents", {})[doc_id] = {
        "source_path": str(pdf_path),
        "source_name": pdf_path.name,
        "sha256": fingerprint,
        "chunk_count": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    registry["embedding_model"] = EMBEDDING_MODEL
    save_registry(registry)
    total = rebuild_bm25(collection)
    print(f"Indexed {pdf_path.name}: {len(chunks)} chunks. Total chunks in index: {total}")
    print(f"Vector store: {CHROMA_DIR}")
    print(f"BM25 index: {BM25_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index PDF files into ChromaDB and BM25.")
    parser.add_argument("--input", help="Path to a PDF to index")
    parser.add_argument("--force", action="store_true", help="Replace existing chunks for this PDF")
    parser.add_argument("--status", action="store_true", help="Show indexed PDFs")
    parser.add_argument("--no-commit", action="store_true", help="Accepted for CLI compatibility; no auto-commit is performed")
    args = parser.parse_args()
    if args.status:
        status(args.input)
        return
    if not args.input:
        parser.error("Provide --input PATH or --status.")
    index_pdf(args.input, args.force)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
