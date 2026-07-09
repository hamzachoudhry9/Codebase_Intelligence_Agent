"""ingest/embed.py - Shared embedding model singleton.

Both agent/tools.py and memory/session_store.py previously loaded their own
130 MB HuggingFace model (BUG-13). This module ensures exactly one instance
is ever created.

Import with:
    from ingest.embed import get_embed_model
"""

import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

_EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
_embed_model = None


def get_embed_model() -> HuggingFaceEmbedding:
    """Return the shared embedding model, loading it on first call."""
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbedding(model_name=_EMBED_MODEL_NAME)
    return _embed_model
