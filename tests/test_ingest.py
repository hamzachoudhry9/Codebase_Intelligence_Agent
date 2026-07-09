"""tests/test_ingest.py - Ingest pipeline tests."""
import pytest
from pathlib import Path

@pytest.fixture
def sample_repo(tmp_path):
    (tmp_path / "mymodule").mkdir()
    (tmp_path / "mymodule/__init__.py").write_text("")
    (tmp_path / "mymodule/auth.py").write_text('''
def verify_token(token: str) -> bool:
    """Verify a JWT token."""
    return bool(token and token.startswith("Bearer "))

def generate_token(user_id: int) -> str:
    """Generate a token for a user."""
    return f"Bearer {user_id}"
''')
    return str(tmp_path)

class TestCodeChunker:
    def test_python_chunked_by_function(self, sample_repo):
        from ingest.code_chunker import chunk_file
        chunks = list(chunk_file(
            str(Path(sample_repo) / "mymodule/auth.py"),
            repo_root=sample_repo
        ))
        assert len(chunks) >= 2
        names = [c["metadata"]["function"] for c in chunks]
        assert "verify_token" in names
        assert "generate_token" in names

    def test_metadata_has_required_fields(self, sample_repo):
        from ingest.code_chunker import chunk_file
        for chunk in chunk_file(
            str(Path(sample_repo) / "mymodule/auth.py"),
            repo_root=sample_repo
        ):
            for field in ("file", "function", "language", "start_line", "end_line"):
                assert field in chunk["metadata"], f"Missing '{field}'"

class TestBenchSelfcheck:
    """The benchmark self-check is the fastest proof the harness works."""
    def test_all_tasks_valid(self):
        from bench.selfcheck import main
        assert main() == 0
