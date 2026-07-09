def parse_kv(text: str) -> dict:
    result = {}
    for seg in text.split(";"):
        seg = seg.strip()
        if not seg or "=" not in seg:
            continue
        key, value = seg.split("=", 1)
        result[key] = value             # MUTANT: no inner strip()
    return result
