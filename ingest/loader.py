"""
loader.py - Load raw documents from the docs directory using LlamaIndex's
SimpleDirectoryReader. Supports .md, .txt, .py, and .rst files recursively.
"""

from llama_index.core import SimpleDirectoryReader, Document
from pathlib import Path


def load_docs(docs_dir: str) -> list[Document]:
    """Load all supported documents from docs_dir recursively.

    Args:
        docs_dir: Path to directory containing documentation files.

    Returns:
        List of LlamaIndex Document objects ready for indexing.

    Raises:
        FileNotFoundError: If docs_dir does not exist.
        ValueError: If no supported files are found.
    """
    path = Path(docs_dir)
    if not path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    reader = SimpleDirectoryReader(
        input_dir=str(path),
        recursive=True,
        required_exts=[".md", ".txt", ".py", ".rst"],
    )

    documents = reader.load_data()

    if not documents:
        raise ValueError(
            f"No supported documents (.md, .txt, .py, .rst) found in {docs_dir}. "
            "Add documentation files before running build_index.py."
        )

    print(f"Loaded {len(documents)} document(s) from {docs_dir}")
    return documents
