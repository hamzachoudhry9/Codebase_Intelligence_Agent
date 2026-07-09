import pytest
from src.retry import retry


def make_flaky(fail_times):
    state = {"n": 0}

    def f():
        if state["n"] < fail_times:
            state["n"] += 1
            raise RuntimeError("transient")
        return 42
    return f


def test_returns_immediately():
    assert retry(make_flaky(0), attempts=3) == 42


def test_succeeds_on_final_attempt():
    assert retry(make_flaky(2), attempts=3, base_delay=0.0) == 42


def test_raises_when_budget_exhausted():
    with pytest.raises(RuntimeError):
        retry(make_flaky(5), attempts=2, base_delay=0.0)
