from typing import List

import pytest

from setml.tasks import IndexTuple, build_m_tuples


def test_index_tuple_is_valid() -> None:
    valid_tuple_1 = IndexTuple(indices=[1, 14], values=[23, 23])
    valid_tuple_2 = IndexTuple(indices=[4, 20], values=[1, 23])

    invalid_tuple_1 = IndexTuple(indices=[1, 1], values=[23, 23])
    invalid_tuple_2 = IndexTuple(indices=[1, 1], values=[1, 23])

    assert valid_tuple_1.is_valid()
    assert valid_tuple_2.is_valid()
    assert not invalid_tuple_1.is_valid()
    assert not invalid_tuple_2.is_valid()


def test_index_tuple_would_be_valid() -> None:
    tuple_1 = IndexTuple(indices=[1, 14], values=[23, 23])

    assert tuple_1.would_be_valid(index=12)
    assert not tuple_1.would_be_valid(index=1)


def test_index_tuple_append() -> None:
    expected_tuple = (2, 78, 42)

    t_1 = IndexTuple(indices=[1, 14], values=list(expected_tuple[:-1]))
    t_1.append(index=15, value=expected_tuple[-1])

    assert t_1.to_tuple() == expected_tuple


def test_index_tuple_to_tuple() -> None:
    expected_tuple = (2, 78)
    t_1 = IndexTuple(indices=[1, 14], values=list(expected_tuple))

    assert t_1.to_tuple() == expected_tuple


def test_index_tuple_to_list() -> None:
    expected_list = [2, 78]
    t_1 = IndexTuple(indices=[1, 14], values=expected_list)

    assert t_1.to_list() == expected_list


@pytest.fixture
def default_values() -> List[int]:
    return [1, 14, -15, 92, -42]


def test_build_m_tuples_singletons(default_values: List[int]) -> None:
    expected_singletons = [(v,) for v in default_values]

    actual_singletons = build_m_tuples(default_values, m=1)

    # We don't care about the order, just that all elements are there.
    assert len(actual_singletons) == len(expected_singletons)
    assert set(actual_singletons) == set(expected_singletons)


def test_build_m_tuples_pairs(default_values: List[int]) -> None:
    expected_pairs = []
    for i, v_i in enumerate(default_values):
        for j, v_j in enumerate(default_values):
            if i != j:
                expected_pairs.append((v_i, v_j))

    actual_pairs = build_m_tuples(default_values, m=2)

    # We don't care about the order, just that all elements are there.
    assert len(actual_pairs) == len(expected_pairs)
    assert set(actual_pairs) == set(expected_pairs)


def test_build_m_tuples_triples(default_values: List[int]) -> None:
    expected_triples = []
    for i, v_i in enumerate(default_values):
        for j, v_j in enumerate(default_values):
            for k, v_k in enumerate(default_values):
                if i != j and i != k and j != k:
                    expected_triples.append((v_i, v_j, v_k))

    actual_triples = build_m_tuples(default_values, m=3)

    # We don't care about the order, just that all elements are there.
    assert len(actual_triples) == len(expected_triples)
    assert set(actual_triples) == set(expected_triples)


def test_build_m_tuples_length_values() -> None:
    values = [-1, 2, 3]
    expected_tuples = [
        (-1, 2, 3),
        (-1, 3, 2),
        (2, -1, 3),
        (2, 3, -1),
        (3, -1, 2),
        (3, 2, -1),
    ]

    actual_tuples = build_m_tuples(values, m=len(values))

    # We don't care about the order, just that all elements are there.
    assert len(actual_tuples) == len(expected_tuples)
    assert set(actual_tuples) == set(expected_tuples)


def test_build_m_tuples_too_few_values() -> None:
    values = [5.0, 42.0, 0.0]
    expected_tuples = []

    actual_tuples = build_m_tuples(values, m=len(values) + 1)

    # We don't care about the order, just that all elements are there.
    assert len(actual_tuples) == len(expected_tuples)
    assert set(actual_tuples) == set(expected_tuples)
