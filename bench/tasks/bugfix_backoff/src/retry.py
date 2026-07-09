import time


def retry(func, attempts: int = 3, base_delay: float = 0.0):
    """Call func up to `attempts` times; return result or raise last error."""
    last_exc = None
    for i in range(attempts - 1):  # BUG: off-by-one -> only attempts-1 tries
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            time.sleep(base_delay * (2 ** i))
    raise last_exc
