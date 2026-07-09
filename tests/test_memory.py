"""tests/test_memory.py - Session memory tests."""

class TestRelevanceThreshold:
    def test_irrelevant_session_filtered(self):
        """BUG-08 regression: unrelated sessions must not be returned."""
        from memory.session_store import SessionMemoryStore
        store = SessionMemoryStore()
        store.save_session(
            query="What is the capital of France?",
            plan=["[search_docs] France"],
            result="Paris", tools_used=["search_docs"],
        )
        results = store.retrieve_relevant_sessions("How do I fix a CUDA OOM error?")
        assert all("France" not in r["summary"] for r in results), \
            "Irrelevant session returned despite threshold"

    def test_pagination(self):
        """BUG-23 regression: sessions endpoint must support offset."""
        from memory.session_store import SessionMemoryStore
        store = SessionMemoryStore()
        for i in range(5):
            store.save_session(f"Query {i}", [f"plan {i}"], f"result {i}", ["search_docs"])
        p1 = store.list_recent_sessions(limit=3, offset=0)
        p2 = store.list_recent_sessions(limit=3, offset=3)
        assert len(p1) <= 3
        if p1 and p2:
            assert not (set(r["summary"] for r in p1) & set(r["summary"] for r in p2))

    def test_count_method(self):
        from memory.session_store import SessionMemoryStore
        store = SessionMemoryStore()
        assert isinstance(store.count(), int)
