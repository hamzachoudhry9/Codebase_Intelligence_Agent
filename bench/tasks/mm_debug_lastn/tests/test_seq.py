from src.seq import last_n


def test_last_two():
    assert last_n([1, 2, 3, 4], 2) == [3, 4]


def test_last_all():
    assert last_n([5, 6], 2) == [5, 6]


def test_zero_is_empty():
    assert last_n([], 0) == []
