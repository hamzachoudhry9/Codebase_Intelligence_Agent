import os
from chromadb import PersistentClient

def health():
    status = {"status": "ok", "version": "2.0.0", "index": {}}

    try:
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        client = PersistentClient(path=chroma_dir)
        docs_coll = client.get_collection("project_docs")
        mem_coll  = client.get_or_create_collection("session_memory")

        status["index"] = {
            "project_docs_chunks": docs_coll.count(),
            "session_memory_sessions": mem_coll.count(),
        }
    except Exception as e:
        status["index"] = {"error": str(e)}

    return status

print(health())