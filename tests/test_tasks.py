import math

import pytest

import torch

from setml import tasks


def test_avg_m_tuple_sum_n_less_m() -> None:
    """
    Test average sum of m tuples with not enough elements in set.

    Result in this case should be just the mean of the set.
    """
    xs = [1, 2]
    y = 1.5

    assert tasks.avg_m_tuple_sum(xs, m=len(xs) + 1) == y


def test_avg_m_tuple_sum_pairs() -> None:
    """Test average sum of pairs."""
    xs_1 = [1, 2, 3, 4, 5]
    y_1 = 6.0

    xs_2 = [5, 10, 7]
    y_2 = 44.0 / 3.0

    xs_3 = [1, 2, -3, -4, 5]
    y_3 = 2.0 / 5.0

    xs_4 = [5, 10, -7]
    y_4 = 16.0 / 3.0

    assert tasks.avg_m_tuple_sum(xs_1, m=2) == y_1, "Length 5, all positive"
    assert tasks.avg_m_tuple_sum(xs_2, m=2) == y_2, "Length 3, all positive"
    assert (
        tasks.avg_m_tuple_sum(xs_3, m=2) == y_3
    ), "Length 5, positive and negative"
    assert (
        tasks.avg_m_tuple_sum(xs_4, m=2) == y_4
    ), "Length 3, positive and negative"


def test_avg_m_tuple_sum_triple() -> None:
    """Test average sum of triples."""
    test_cases = {
        "Length 5, all positive": [1, 2, 3, 4, 5],
        "Length 4, all positive": [5, 10, 7, 2],
        "Length 5, positive and negative": [1, 2, -3, -4, 5],
        "Length 4, positive and negative": [5, 10, -7, 3],
    }

    for description, test_case in test_cases.items():
        triple_sum = 0.0
        n_triple = 0
        for i, x_i in enumerate(test_case):
            for j, x_j in enumerate(test_case):
                for k, x_k in enumerate(test_case):
                    if i != j and j != k and i != k:
                        triple_sum += x_i + x_j + x_k
                        n_triple += 1
        actual = tasks.avg_m_tuple_sum(test_case, m=3)
        expected = triple_sum / n_triple
        assert math.isclose(actual, expected), description


@pytest.fixture
def avg_pair_sum() -> tasks.AverageMTupleSumTask:
    return tasks.AverageMTupleSumTask(2)


def test_avg_pair_sum_init(avg_pair_sum: tasks.AverageMTupleSumTask) -> None:
    assert avg_pair_sum.m == 2
    assert avg_pair_sum.label == "average_pair_sum"
    assert avg_pair_sum.loss == "mse"
    assert avg_pair_sum.loss_fn == torch.nn.functional.mse_loss


def test_avg_pair_sum_padding_element(
    avg_pair_sum: tasks.AverageMTupleSumTask,
) -> None:
    assert avg_pair_sum.padding_element == 0
