from src.cache import LRUCache


def test_basic_get_put():
    c = LRUCache(2)
    c.put("a", 1)
    assert c.get("a") == 1


def test_get_marks_recently_used():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1      # touch 'a' -> 'b' is now least recent
    c.put("c", 3)               # should evict 'b', keep 'a'
    assert c.get("a") == 1
    assert c.get("b") is None


def test_capacity_one_eviction():
    c = LRUCache(1)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") is None
    assert c.get("b") == 2
