# document-reader-chromaDb (v3)

A local PDF question-answering project using ChromaDB, BGE embeddings, BM25 hybrid retrieval, reciprocal-rank fusion, and a cross-encoder reranker.

## Features
- Index PDFs on demand; supports multiple documents.
- Persistent ChromaDB vector store and a rebuilt BM25 lexical index.
- 300-word chunks with 50-word overlap and page metadata.
- Hybrid retrieval: vector top 50 + BM25 top 50 -> RRF -> rerank top 40.
- Optional identifier boost for queries containing code-like identifiers.
- Answers grounded in retrieved excerpts, with source document and page citations.
- Optional C++ source-tree search via ripgrep (or a Python fallback).

## Requirements
- Python 3.10 or newer
- Windows, macOS, or Linux
- About 2 GB+ disk for embedding/reranker model caches. The optional answer model needs additional space and RAM.
- For scanned/image-only PDFs, add OCR separately; this version extracts embedded text.

## Setup (Windows PowerShell)
```powershell
cd "$HOME\Desktop\Personal_projects"
Expand-Archive .\document-reader-chromadb-v3.zip -DestinationPath .
cd .\document-reader-chromadb-v3
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The first run downloads models from Hugging Face. They are cached locally.

## Index a PDF
```powershell
python scripts\index.py --input "C:\path\to\document.pdf"
python scripts\index.py --status
python scripts\index.py --input "C:\path\to\document.pdf" --force
```

To index a second PDF, run `index.py --input` again. It is added to the same persistent store. Re-indexing a file requires `--force`.

## Ask questions
```powershell
python scripts\query.py --q "What does the document say about repeatable subforms?" --k 5
python scripts\query.py --q "WordNum Spanish" --k 5 --answer
```

Retrieval-only output is the default. `--answer` asks a local instruction model to compose an answer from the retrieved context. The default is `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0); first use downloads several GB and may be slow. Override with `--llm-model MODEL_ID` or `DOCUMENT_READER_LLM_MODEL`. You can also use `--no-answer` to make intent explicit.

If the answer model does not fit your machine, use retrieval-only mode or configure a compatible local model. The embedding and reranking models are separate from the generation model.

## C++ code search
```powershell
python scripts\search_code.py --root "C:\path\to\cpp\repo" --symbol "SubformManager"
python scripts\search_code.py --root "C:\path\to\cpp\repo" --grep "XFA_REPEAT"
python scripts\scripts\search_code.py --root "C:\path\to\cpp\repo" --read src\renderer\XFASubform.cpp --lines 200 350
```
Use `--root` to point at the C++ repository. If omitted, the script searches four levels above itself.

## Data layout
```text
data/
  chroma_store/             # persistent vector store (generated locally)
  bm25_index.pkl             # lexical index (generated locally)
  multi_doc_registry.json    # indexed-document registry
```
PDF source files are not copied into the project. Index artefacts are local and are excluded from Git by default because they may contain proprietary document text.

## Design notes
- Registry stores the absolute source path, content fingerprint, chunk count, and model name.
- Each chunk has `doc_id`, `source_name`, `source_path`, `start_page`, `end_page`, and chunk index metadata.
- ChromaDB's default cosine distance is used. The returned `similarity` is `1 - distance`.
- BM25 is rebuilt from all stored chunks after indexing. Pickle files should only be loaded from trusted local sources.
- RRF uses `1 / (60 + rank)` for each ranked list. An identifier-shaped query receives a 1.5x BM25 contribution.
- The generator is instructed to answer only from context and say when the context is insufficient. Always verify citations against the source document.

## Notes
- Git LFS is not required for a fresh personal repo because indexes are generated locally. Add Git LFS only if you intentionally want to version large index artefacts.
- Do not commit confidential PDFs or derived indexes containing confidential text.
