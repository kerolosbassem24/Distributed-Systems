import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VECTOR_DB_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model

def _extract_text(filepath: str) -> str:
    if filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def _chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 50]

def ingest_documents(docs_folder: str) -> None:
    client     = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = client.get_or_create_collection("knowledge_base")

    files = [f for f in os.listdir(docs_folder)
             if f.endswith(".txt") or f.endswith(".pdf")]
    total_chunks = 0

    embed_model = _get_embed_model()

    for filename in files:
        path   = os.path.join(docs_folder, filename)
        text   = _extract_text(path)
        chunks = _chunk_text(text)

        if not chunks:
            continue

        embeddings = embed_model.encode(chunks, show_progress_bar=False).tolist()

        ids      = [f"{filename}__chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": i}
                     for i in range(len(chunks))]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)
        print(f"  [OK] {filename}: {len(chunks)} chunks ingested")

    print(f"\n[RAG] Total: {total_chunks} chunks across {len(files)} documents.")

def is_db_populated() -> bool:
    try:
        client     = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        collection = client.get_or_create_collection("knowledge_base")
        return collection.count() > 0
    except Exception:
        return False
