import math

import pytest

import torch

from setml.tasks import (
    DesperateStudentMTupleTask,
    decaying_avg_of_inv_exponentials,
)

###########
# Utility #
###########


def test_decaying_avg_of_inv_exponentials() -> None:
    """
    Formula for y:
        y = 1 / m sum_{j=1}^m alpha^{j - 1} * sigma * exp(-x_j / sigma)
    """
    xs_1 = [2.0, -1.0]
    expected_y_1 = 4.575195
    actual_y_1 = decaying_avg_of_inv_exponentials(
        xs_1, alpha=0.99, sigma=100.0
    )

    xs_2 = [-20.0, 3.0, 150.0, 0.0]
    expected_y_2 = 1.006965
    actual_y_2 = decaying_avg_of_inv_exponentials(xs_2, alpha=0.9, sigma=10.0)

    assert math.isclose(actual_y_1, expected_y_1, rel_tol=1e-6)
    assert math.isclose(actual_y_2, expected_y_2, rel_tol=1e-6)


##########################
# Desperate Student Pair #
##########################


@pytest.fixture
def desperate_student_pair() -> DesperateStudentMTupleTask:
    return DesperateStudentMTupleTask(m=2, alpha=0.99, sigma=100.0)


def test_desperate_student_pair_init(
    desperate_student_pair: DesperateStudentMTupleTask,
) -> None:
    assert desperate_student_pair.label == "desperate_student_2_tuple"
    assert desperate_student_pair.loss == "mse"
    assert desperate_student_pair.loss_fn == torch.nn.functional.mse_loss


def test_desperate_student_pair_permut_invariant(
    desperate_student_pair: DesperateStudentMTupleTask,
) -> None:
    x_1 = torch.tensor([3, 4, -1, 0]).float().unsqueeze(-1)
    x_2 = torch.tensor([-1, 4, 3, 0]).float().unsqueeze(-1)

    assert desperate_student_pair.generate_label(
        x_1
    ) == desperate_student_pair.generate_label(x_2)


def test_desperate_student_pair_too_few_values(
    desperate_student_pair: DesperateStudentMTupleTask,
) -> None:
    xs = torch.tensor([3.0]).float().unsqueeze(-1)
    expected_label = torch.tensor(4.556170).float()
    actual_label = desperate_student_pair.generate_label(xs)

    assert torch.allclose(actual_label, expected_label)
