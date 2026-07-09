def last_n(items, n):
    """Return the last n items of `items`, in their original order."""
    result = []
    for i in range(n):
        result.append(items[len(items) - i])  # BUG: off-by-one -> IndexError
    return list(reversed(result))
