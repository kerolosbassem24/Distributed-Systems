import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VECTOR_DB_PATH, EMBEDDING_MODEL, TOP_K_CHUNKS

_client     = None
_collection = None
_embed_model = None

import threading
_embed_lock = threading.Lock()

def _get_embed_model():
    global _embed_model
    with _embed_lock:
        if _embed_model is None:
            _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model

def _get_collection():
    global _client, _collection
    if _collection is None:
        _client     = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        _collection = _client.get_or_create_collection("knowledge_base")
    return _collection

def retrieve_context(query: str) -> str:
    collection     = _get_collection()
    embed_model    = _get_embed_model()
    query_embedding = embed_model.encode([query]).tolist()
    results        = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K_CHUNKS,
        include=["documents", "metadatas", "distances"],
    )
    chunks = results["documents"][0] if results["documents"] else []
    return "\n\n".join(chunks) if chunks else "No relevant context found."
