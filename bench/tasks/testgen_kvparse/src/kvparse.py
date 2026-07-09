def parse_kv(text: str) -> dict:
    """Parse 'a=1; b=2' into {'a': '1', 'b': '2'}.

    - segments separated by ';'
    - each segment split on the FIRST '='
    - keys and values are whitespace-stripped
    - empty or '='-less segments are skipped
    """
    result = {}
    for seg in text.split(";"):
        seg = seg.strip()
        if not seg or "=" not in seg:
            continue
        key, value = seg.split("=", 1)
        result[key.strip()] = value.strip()
    return result
