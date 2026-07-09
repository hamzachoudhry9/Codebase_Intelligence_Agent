import re


def slugify(text: str) -> str:
    """Lowercase, replace non-alphanumeric runs with a single hyphen."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def truncate_words(text: str, n: int) -> str:
    """Return the first n whitespace-separated words of text."""
    words = text.split()
    return " ".join(words[:n])
