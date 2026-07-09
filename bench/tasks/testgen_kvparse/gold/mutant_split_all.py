def parse_kv(text: str) -> dict:
    result = {}
    for seg in text.split(";"):
        seg = seg.strip()
        if not seg or "=" not in seg:
            continue
        parts = seg.split("=")          # MUTANT: split on every '='
        key, value = parts[0], parts[1]
        result[key.strip()] = value.strip()
    return result
