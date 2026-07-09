from src.textutil import slugify, truncate_words


def test_basic_slug():
    assert slugify("Hello World") == "hello-world"


def test_trailing_punct_no_dangling_hyphen():
    assert slugify("Hello, World!") == "hello-world"


def test_leading_whitespace_stripped():
    assert slugify("  spaces ") == "spaces"


def test_truncate_words_unchanged():
    assert truncate_words("the quick brown fox", 2) == "the quick"
