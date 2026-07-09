"""tests/conftest.py - Shared fixtures for all tests."""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="session")
def fake_chroma_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("chroma"))

@pytest.fixture(autouse=True)
def set_env(monkeypatch, fake_chroma_dir):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", fake_chroma_dir)
    monkeypatch.setenv("AGENT_API_KEY", "test-key-12345")
    monkeypatch.setenv("DISABLE_AUTH", "true")
    monkeypatch.setenv("AGENT_MODEL", "llama3.1:8b")
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
