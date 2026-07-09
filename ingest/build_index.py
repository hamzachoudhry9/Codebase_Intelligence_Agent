"""
ingest/build_index.py - Production knowledge base builder.

Three ChromaDB collections:
  project_docs   - all .py, .cpp, .cu, .h, .md files chunked at function level
  session_memory - created automatically by SessionMemoryStore (not touched here)
  stackoverflow  - scraped Q&A pairs (optional, cached after first run)

Usage:
  python ingest/build_index.py                         # default: repo=. docs=docs/
  python ingest/build_index.py --repo /path/to/repo    # different codebase
  python ingest/build_index.py --scrape-so             # also scrape Stack Overflow
  python ingest/build_index.py --no-wipe               # append, don't rebuild
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ingest.code_chunker import chunk_file, iter_repo_files

load_dotenv()

_EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_DOCS_COLLECTION  = "project_docs"
_SO_COLLECTION    = "stackoverflow"
_BATCH_SIZE       = 50

SO_TAGS = [
    "python+langchain",
    "python+chromadb",
    "python+langgraph",
    "cuda+out-of-memory",
    "cuda+kernel",
    "python+llm",
    "python+rag",
    "python+fastapi",
]

SO_CACHE_PATH = Path("data/stackoverflow_qa.json")


# ── Stack Overflow scraping ────────────────────────────────────────────────

def scrape_stackoverflow(tags: list[str], questions_per_tag: int = 25) -> list[dict]:
    """Scrape SO via the Stack Exchange API (no key, 300 req/day free)."""
    try:
        import requests as req_lib
    except ImportError:
        print("  requests not installed - skipping SO scrape")
        return []

    qa_pairs: list[dict] = []
    base_url = "https://api.stackexchange.com/2.3"
    seen_ids: set = set()

    for tag in tags:
        print(f"  Scraping SO tag: {tag}")
        try:
            r = req_lib.get(
                f"{base_url}/questions",
                params={"order": "desc", "sort": "votes", "tagged": tag,
                        "site": "stackoverflow", "filter": "withbody",
                        "pagesize": questions_per_tag},
                timeout=15,
            )
            if r.status_code != 200:
                continue
            for q in r.json().get("items", []):
                qid = q.get("question_id")
                if qid in seen_ids or not q.get("is_answered"):
                    continue
                seen_ids.add(qid)
                ar = req_lib.get(
                    f"{base_url}/questions/{qid}/answers",
                    params={"order": "desc", "sort": "votes", "site": "stackoverflow",
                            "filter": "withbody", "pagesize": 1},
                    timeout=15,
                )
                if ar.status_code != 200:
                    continue
                answers = ar.json().get("items", [])
                if not answers:
                    continue
                strip = lambda h: re.sub(r"<[^>]+>", " ", h).strip()
                qa_pairs.append({
                    "question": f"{q.get('title','')}\n\n{strip(q.get('body',''))[:800]}",
                    "answer": strip(answers[0].get("body", ""))[:1200],
                    "url": q.get("link", ""),
                    "tags": q.get("tags", []),
                    "score": q.get("score", 0),
                })
                time.sleep(0.1)
        except Exception as e:
            print(f"    Error on {tag}: {e}")
        time.sleep(1)

    print(f"  Scraped {len(qa_pairs)} Q&A pairs")
    return qa_pairs


def load_or_scrape_so(tags: list[str], force: bool = False) -> list[dict]:
    if SO_CACHE_PATH.exists() and not force:
        print(f"  Loading cached SO data from {SO_CACHE_PATH}")
        with open(SO_CACHE_PATH) as f:
            return json.load(f)
    SO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pairs = scrape_stackoverflow(tags)
    with open(SO_CACHE_PATH, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"  Saved {len(pairs)} pairs to {SO_CACHE_PATH}")
    return pairs


# ── Error log ingestion ────────────────────────────────────────────────────

def load_error_logs(error_log_dir: str = "./data/error_logs") -> list[dict]:
    chunks: list[dict] = []
    log_dir = Path(error_log_dir)
    if not log_dir.exists():
        print(f"  No error_logs dir at {log_dir} - skipping (create data/error_logs/*.txt to add)")
        return chunks
    for path in list(log_dir.glob("*.txt")) + list(log_dir.glob("*.log")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if len(text) < 30:
                continue
            chunks.append({
                "text": text[:3000],
                "metadata": {
                    "file": str(path), "function": path.stem,
                    "type": "error_log", "language": "log",
                    "start_line": 1, "end_line": len(text.splitlines()), "docstring": "",
                },
            })
        except Exception:
            continue
    print(f"  Loaded {len(chunks)} error log files")
    return chunks


# ── Embedding + ChromaDB insertion ─────────────────────────────────────────

def embed_and_insert(
    embed_model: HuggingFaceEmbedding,
    collection: chromadb.Collection,
    chunks: list[dict],
    id_prefix: str = "chunk",
) -> int:
    if not chunks:
        return 0
    inserted = 0
    for batch_start in range(0, len(chunks), _BATCH_SIZE):
        batch     = chunks[batch_start: batch_start + _BATCH_SIZE]
        texts     = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        ids       = [f"{id_prefix}_{batch_start + i:06d}" for i in range(len(batch))]
        embeddings = embed_model.get_text_embedding_batch(texts, show_progress=False)
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        inserted += len(batch)
        print(f"    {inserted}/{len(chunks)} embedded ({inserted*100//len(chunks)}%)", end="\r")
    print()
    return inserted


# ── Main ───────────────────────────────────────────────────────────────────

def build_index(
    repo_root: str = ".",
    extra_docs_dir: str | None = None,
    wipe: bool = True,
    scrape_so: bool = False,
    force_scrape: bool = False,
) -> dict:
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    print(f"\n{'='*60}")
    print("Codebase Intelligence Agent - Knowledge Base Builder")
    print(f"{'='*60}")
    print(f"Repo root : {Path(repo_root).resolve()}")
    print(f"ChromaDB  : {Path(chroma_dir).resolve()}")
    print()

    print(f"Loading embedding model ({_EMBED_MODEL_NAME})...")
    t0 = time.time()
    embed_model = HuggingFaceEmbedding(model_name=_EMBED_MODEL_NAME)
    print(f"  Ready in {time.time()-t0:.1f}s\n")

    client = chromadb.PersistentClient(path=chroma_dir)

    # ── project_docs ───────────────────────────────────────────────────────
    print(f"[1/2] Building '{_DOCS_COLLECTION}'...")
    if wipe:
        try:
            client.delete_collection(_DOCS_COLLECTION)
            print("  Deleted existing collection")
        except Exception:
            pass

    docs_collection = client.get_or_create_collection(_DOCS_COLLECTION)

    source_exts = (".py", ".cpp", ".cc", ".c", ".h", ".hpp", ".cu", ".cuh")
    doc_exts    = (".md", ".rst", ".txt")
    all_exts    = source_exts + doc_exts

    files: list[str] = list(iter_repo_files(repo_root, extensions=all_exts))
    print(f"  Found {len(files)} files in {repo_root}")

    if extra_docs_dir and extra_docs_dir != repo_root:
        extra_path = Path(extra_docs_dir)
        if extra_path.exists():
            existing = set(files)
            extra = [f for f in iter_repo_files(extra_docs_dir, extensions=doc_exts)
                     if f not in existing]
            files.extend(extra)
            print(f"  Found {len(extra)} extra files in {extra_docs_dir}")

    all_chunks: list[dict] = []
    by_lang: dict[str, int] = {}
    for fp in files:
        for c in chunk_file(fp, repo_root=repo_root):
            lang = c["metadata"].get("language", "unknown")
            by_lang[lang] = by_lang.get(lang, 0) + 1
            all_chunks.append(c)

    # Add error logs
    error_chunks = load_error_logs()
    all_chunks.extend(error_chunks)

    print(f"  Total chunks: {len(all_chunks)}")
    for lang, n in sorted(by_lang.items()):
        print(f"    {lang}: {n} chunks")
    print("  Embedding...")

    embed_and_insert(embed_model, docs_collection, all_chunks, "doc")
    print(f"  '{_DOCS_COLLECTION}' → {docs_collection.count()} chunks\n")

    # ── stackoverflow ──────────────────────────────────────────────────────
    so_count = 0
    if scrape_so:
        print(f"[2/2] Building '{_SO_COLLECTION}'...")
        try:
            client.delete_collection(_SO_COLLECTION)
        except Exception:
            pass
        so_collection = client.get_or_create_collection(_SO_COLLECTION)
        pairs = load_or_scrape_so(SO_TAGS, force=force_scrape)
        so_chunks = [{
            "text": f"Question: {p['question']}\n\nAnswer: {p['answer']}"[:3000],
            "metadata": {
                "file": p.get("url", ""), "function": p["question"][:80],
                "type": "stackoverflow", "language": "qa",
                "start_line": 0, "end_line": 0, "docstring": "",
                "score": p.get("score", 0),
            },
        } for p in pairs]
        so_count = embed_and_insert(embed_model, so_collection, so_chunks, "so")
        print(f"  '{_SO_COLLECTION}' → {so_collection.count()} chunks\n")
    else:
        print("[2/2] Stack Overflow skipped (add --scrape-so to enable)\n")

    # ── Summary + sample ───────────────────────────────────────────────────
    print(f"{'='*60}")
    print("Build complete!")
    print(f"  project_docs : {docs_collection.count()} chunks")
    if scrape_so:
        print(f"  stackoverflow: {so_count} chunks")
    print()
    print("Sample chunks:")
    sample = docs_collection.peek(limit=4)
    for doc, meta in zip(sample["documents"], sample["metadatas"]):
        loc = f"{meta.get('file','?')}:{meta.get('function','?')}"
        preview = doc[:70].replace("\n", " ")
        print(f"  [{meta.get('language','?'):8s}] {loc} - {preview}...")
    print()

    return {"project_docs": docs_collection.count(), "stackoverflow": so_count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Codebase Intelligence Agent knowledge base")
    parser.add_argument("--repo",        default=".",    help="Repository root to ingest")
    parser.add_argument("--docs",        default=None,   help="Extra docs directory")
    parser.add_argument("--no-wipe",     action="store_true", help="Append instead of rebuild")
    parser.add_argument("--scrape-so",   action="store_true", help="Scrape Stack Overflow Q&A")
    parser.add_argument("--force-scrape",action="store_true", help="Re-scrape even if cached")
    args = parser.parse_args()

    build_index(
        repo_root=args.repo,
        extra_docs_dir=args.docs,
        wipe=not args.no_wipe,
        scrape_so=args.scrape_so,
        force_scrape=args.force_scrape,
    )
