from src.kvparse import parse_kv


def test_basic():
    assert parse_kv("a=1; b=2") == {"a": "1", "b": "2"}


def test_value_with_equals():
    # kills mutant_split_all
    assert parse_kv("expr=a=b") == {"expr": "a=b"}


def test_strips_whitespace():
    # kills mutant_no_strip
    assert parse_kv("  a = 1 ") == {"a": "1"}


def test_skips_empty_and_malformed():
    assert parse_kv("a=1;; junk ; b=2") == {"a": "1", "b": "2"}
