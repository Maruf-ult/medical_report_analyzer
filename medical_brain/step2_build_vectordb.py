"""
MEDICAL BRAIN - Step 2: Build Vector Database
===============================================
Reads all processed .txt files and loads them into ChromaDB.
This gives the AI "memory" of all 2,000 reports.

HOW TO RUN:
    python step2_build_vectordb.py

WHAT IT DOES:
    - Reads every .txt file from processed_text/
    - Splits them into chunks (paragraphs)
    - Converts chunks to embeddings (numbers the AI understands)
    - Stores everything in a local ChromaDB database
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Windows UTF-8 Fix
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Paths
PROCESSED_DIR = Path("processed_text")
DB_DIR        = Path("vector_db")
LOG_DIR       = Path("logs")
DB_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Logging
log_file = LOG_DIR / f"step2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 400   # characters per chunk
CHUNK_OVERLAP = 80    # overlap between chunks so context isn't lost
COLLECTION    = "medical_reports"

# ── Text Chunker ──────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str):
    """
    Split text into overlapping chunks.
    Each chunk gets metadata so we know which report it came from.
    """
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        # Try to break at a newline or period instead of mid-sentence
        if end < len(text):
            for break_char in ['\n', '. ', ' ']:
                pos = text.rfind(break_char, start, end)
                if pos > start + (CHUNK_SIZE // 2):
                    end = pos + 1
                    break

        chunk = text[start:end].strip()
        if len(chunk) > 30:  # skip tiny meaningless chunks
            chunks.append({
                "text": chunk,
                "source": source,
                "chunk_index": chunk_index,
                "id": f"{source}_chunk_{chunk_index}"
            })
            chunk_index += 1

        start = end - CHUNK_OVERLAP

    return chunks

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  MEDICAL BRAIN - Step 2: Building Vector Database")
    log.info("=" * 60)

    # Find all processed text files
    txt_files = list(PROCESSED_DIR.glob("*.txt"))
    if not txt_files:
        log.error(f"No .txt files found in '{PROCESSED_DIR}/'")
        log.error("Please run step1_prepare_data.py first!")
        return

    log.info(f"Found {len(txt_files)} processed reports to index.\n")

    # ── Load ChromaDB ─────────────────────────────────────────────────────────
    log.info("[DB] Initializing ChromaDB...")
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Use a free local embedding model (downloads ~90MB once)
    log.info("[DB] Loading embedding model (downloads ~90MB on first run)...")
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    log.info("[DB] Embedding model ready.")

    # Get or create the collection (like a table in a database)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embedder,
        metadata={"description": "Medical reports knowledge base"}
    )

    existing = collection.count()
    log.info(f"[DB] Collection '{COLLECTION}' has {existing} existing chunks.\n")

    # ── Process Each File ─────────────────────────────────────────────────────
    total_chunks = 0
    skipped      = 0
    start_time   = time.time()

    for i, txt_path in enumerate(txt_files, 1):
        source = txt_path.stem
        log.info(f"[{i}/{len(txt_files)}] Indexing: {txt_path.name}")

        # Read the file
        text = txt_path.read_text(encoding='utf-8', errors='replace')

        # Chunk it
        chunks = chunk_text(text, source)
        if not chunks:
            log.warning(f"  [WARN] No chunks from: {txt_path.name}")
            continue

        # Check if already indexed (avoid duplicates on re-run)
        first_id = chunks[0]["id"]
        existing_check = collection.get(ids=[first_id])
        if existing_check["ids"]:
            log.info(f"  [SKIP] Already indexed: {txt_path.name}")
            skipped += 1
            continue

        # Add to ChromaDB
        collection.add(
            ids       = [c["id"]     for c in chunks],
            documents = [c["text"]   for c in chunks],
            metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]
        )

        total_chunks += len(chunks)
        log.info(f"  [OK] Added {len(chunks)} chunks from {txt_path.name}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    final_count = collection.count()

    log.info("\n" + "=" * 60)
    log.info(f"  [OK] Indexing complete!")
    log.info(f"  [CHUNKS] Added {total_chunks} new chunks this run")
    log.info(f"  [TOTAL]  {final_count} total chunks in database")
    log.info(f"  [SKIP]   {skipped} files already indexed")
    log.info(f"  [TIME]   {elapsed:.1f} seconds")
    log.info(f"  [DB]     {DB_DIR.resolve()}")
    log.info("=" * 60)

    # Quick test search
    log.info("\n[TEST] Running a quick search to verify the database...")
    test_results = collection.query(
        query_texts=["blood test CBC hemoglobin"],
        n_results=min(3, final_count)
    )
    if test_results["documents"][0]:
        log.info("[TEST] Search working! Sample result:")
        log.info(f"  -> {test_results['documents'][0][0][:120]}...")
    else:
        log.warning("[TEST] Search returned no results.")

    log.info("\n[DONE] Step 2 complete! Next: step3_setup_ollama.py\n")

if __name__ == "__main__":
    main()