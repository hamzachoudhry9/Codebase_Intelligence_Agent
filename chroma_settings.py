import chromadb

def get_chroma_client(path: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=path,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
