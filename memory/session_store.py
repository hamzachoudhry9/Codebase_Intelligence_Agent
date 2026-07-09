import json, os, uuid
from datetime import datetime
from chroma_settings import get_chroma_client
from dotenv import load_dotenv

load_dotenv()

_store_instance = None

def _get_embed_model():
    """Use the shared singleton from ingest.embed (BUG-13: eliminates 2nd 130MB load)."""
    from ingest.embed import get_embed_model
    return get_embed_model()

def get_session_store():
    global _store_instance
    if _store_instance is None:
        _store_instance = SessionMemoryStore()
    return _store_instance

class SessionMemoryStore:
    COLLECTION_NAME = "session_memory"

    def __init__(self):
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.client = get_chroma_client(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(self.COLLECTION_NAME)
        self.embed_model = _get_embed_model()

    def save_session(self, query, plan, result, tools_used):
        session_id = str(uuid.uuid4())
        summary = f"Query: {query}\nPlan: {'; '.join(plan)}\nResult: {result[:500]}"
        embedding = self.embed_model.get_text_embedding(summary)
        self.collection.add(
            ids=[session_id],
            embeddings=[embedding],
            documents=[summary],
            metadatas=[{
                "query": query,
                "tools_used": json.dumps(tools_used),
                "timestamp": datetime.utcnow().isoformat(),
                "plan_steps": len(plan),
            }],
        )
        return session_id

    _RELEVANCE_THRESHOLD = float(os.getenv("MEMORY_RELEVANCE_THRESHOLD", "0.35"))

    def retrieve_relevant_sessions(self, query, top_k=3):
        """Return past sessions similar to query, filtered by relevance threshold (BUG-08)."""
        count = self.collection.count()
        if count == 0:
            return []
        embedding = self.embed_model.get_text_embedding(query)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        sessions = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if dist <= self._RELEVANCE_THRESHOLD:
                sessions.append({"summary": doc, "metadata": meta, "distance": dist})
        return sessions

    def count(self) -> int:
        return self.collection.count()

    def list_recent_sessions(self, limit: int = 20, offset: int = 0) -> list:
        """Paginated session list (BUG-23 fix).

        NOTE: `peek()` has no offset support and always returns the same
        leading slice, so pagination previously returned identical pages
        regardless of `offset`. `get()` supports `limit`/`offset` directly.
        """
        count = self.collection.count()
        if count == 0 or offset >= count:
            return []
        results = self.collection.get(
            limit=limit, offset=offset, include=["documents", "metadatas"]
        )
        return [
            {"summary": doc, "metadata": meta}
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
