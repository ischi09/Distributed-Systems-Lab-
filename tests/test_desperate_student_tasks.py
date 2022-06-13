import pytest

import torch

from setml import tasks


@pytest.fixture
def desperate_student_pair() -> tasks.DesperateStudentPairTask:
    return tasks.DesperateStudentPairTask()


def test_desperate_student_pair_init(
    desperate_student_pair: tasks.DesperateStudentPairTask,
) -> None:
    assert desperate_student_pair.label == "desperate_student_pair"
    assert desperate_student_pair.loss == "mse"
    assert desperate_student_pair.loss_fn == torch.nn.functional.mse_loss


def test_desperate_student_pair_permut_invariant(
    desperate_student_pair: tasks.DesperateStudentPairTask,
) -> None:
    x_1 = torch.tensor([3, 4, -1, 0], dtype=float).unsqueeze(-1)
    x_2 = torch.tensor([-1, 4, 3, 0], dtype=float).unsqueeze(-1)

    assert desperate_student_pair.generate_label(
        x_1
    ) == desperate_student_pair.generate_label(x_2)
