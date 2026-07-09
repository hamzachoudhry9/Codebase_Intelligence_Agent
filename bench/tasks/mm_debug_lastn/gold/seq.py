def last_n(items, n):
    """Return the last n items of `items`, in their original order."""
    result = []
    for i in range(n):
        result.append(items[len(items) - 1 - i])
    return list(reversed(result))
