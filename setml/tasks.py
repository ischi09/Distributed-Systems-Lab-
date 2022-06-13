from dataclasses import dataclass
import math
import copy
from typing import Any, Callable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import Task as TaskConfig


class Task:
    def __init__(
        self,
        label: str,
        loss: str,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        self.label = label
        self.loss = loss
        self.loss_fn = loss_fn

    def generate_label(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def padding_element(self) -> int:
        """
        Return element to be used as set padding.

        Note: This element ideally should *not* change the label of the set.
        """
        raise NotImplementedError


class RegressionTask(Task):
    def __init__(self, label: str) -> None:
        super().__init__(label=label, loss="mse", loss_fn=F.mse_loss)


class ClassificationTask(Task):
    def __init__(self, label: str, n_classes: int) -> None:
        super().__init__(label=label, loss="ce", loss_fn=F.cross_entropy)

        self.n_classes = n_classes


class MaxTask(RegressionTask):
    def __init__(self) -> None:
        super().__init__(label="max")

    def generate_label(self, x: torch.Tensor) -> torch.Tensor:
        return x.max()

    # TODO: Fix and make it dependent on the task configuration. Should be
    # min. possible element, maybe minus 1.
    @property
    def padding_element(self) -> int:
        return -1


class ModeTask(ClassificationTask):
    def __init__(self, n_classes: int) -> None:
        super().__init__(label="mode", n_classes=n_classes)

    def generate_label(self, x: torch.Tensor) -> torch.Tensor:
        mode = int(x.squeeze().mode().values)
        label = torch.zeros(self.n_classes)
        label[mode] = 1.0
        return label

    # TODO: Fix and make it dependent on the task configuration. Should be
    # random element outside of possible elements.
    @property
    def padding_element(self) -> int:
        return -1


class CardinalityTask(RegressionTask):
    def __init__(self) -> None:
        super().__init__(label="cardinality")

    def generate_label(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(len(x), dtype=torch.float)

    @property
    def padding_element(self) -> int:
        return 0


class SumTask(RegressionTask):
    def __init__(self) -> None:
        super().__init__(label="sum")

    def generate_label(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()

    @property
    def padding_element(self) -> int:
        return 0


class MeanTask(RegressionTask):
    def __init__(self) -> None:
        super().__init__(label="mean")

    def generate_label(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean()

    @property
    def padding_element(self) -> int:
        return 0


class LongestSeqLenTask(RegressionTask):
    def __init__(self) -> None:
        super().__init__(label="longest_seq_length")

    def generate_label(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns length of longest sequence of consecutive numbers.

        Example:
            [2, 4, 1, 5, 7] -> 2 (because 1,2 or 4,5 are consecutive)
        """
        sorted_ = sorted(x.flatten().tolist())
        max_length = 1
        cur_length = 1
        last_val = None
        for val in sorted_:
            if last_val is None:
                last_val = val
                continue

            if last_val + 1 == val or last_val == val:
                cur_length += 1

            else:
                max_length = max(max_length, cur_length)
                cur_length = 1

            last_val = val

        max_length = max(max_length, cur_length)
        return torch.tensor(max_length, dtype=torch.float)

    # TODO: Fix and make it dependent on the task configuration. Should be
    # random element outside of possible elements.
    @property
    def padding_element(self) -> int:
        return -1


class LargestContiguousSumTask(RegressionTask):
    def __init__(self) -> None:
        super().__init__(label="largest_contiguous_sum")

    def generate_label(self, x: torch.Tensor) -> torch.Tensor:
        values = x.flatten().tolist()
        max_val = max(values)

        result = max_val if max_val < 0 else sum([max(v, 0) for v in values])
        return torch.tensor(result, dtype=torch.float)

    @property
    def padding_element(self) -> int:
        return -1


class LargestNTupleSumTask(RegressionTask):
    def __init__(self, n: int) -> None:
        if n == 2:
            label = "largest_pair_sum"
        elif n == 3:
            label = "largest_triple_sum"
        else:
            label = f"largest_{n}_tuple_sum"

        super().__init__(label=label)

        self.n = n

    def generate_label(self, x: torch.Tensor) -> torch.Tensor:
        sorted_, _ = torch.sort(x, 0, descending=True)
        return sorted_[: self.n].sum()

    # TODO: Fix and make it dependent on the task configuration. Should be
    # min. possible element, maybe minus 1.
    @property
    def padding_element(self) -> int:
        return -1


def avg_m_tuple_sum(xs: Sequence[float], m: int) -> float:
    """Compute the average of all sums of m-tuples.

    Note that if the number of elements is less than m, the mean is returned.
    """
    mean = math.fsum(xs) / float(len(xs))
    if len(xs) < m:
        return mean
    return m * mean


class AverageMTupleSumTask(RegressionTask):
    def __init__(self, m: int) -> None:
        if m == 2:
            label = "average_pair_sum"
        elif m == 3:
            label = "average_triple_sum"
        else:
            label = f"average_{m}_tuple_sum"

        super().__init__(label=label)

        self.m = m

    def generate_label(self, x: torch.Tensor) -> torch.Tensor:
        label = avg_m_tuple_sum(x.float().flatten().tolist(), self.m)
        return torch.tensor(label)

    @property
    def padding_element(self) -> int:
        return 0


class ContainsEvenTask(ClassificationTask):
    def __init__(self) -> None:
        super().__init__(label="mode", n_classes=2)

    def generate_label(self, x: torch.Tensor) -> torch.Tensor:
        values = x.flatten().tolist()
        contains_even = any([v % 2 == 0 for v in values])
        label = [0, 1] if contains_even else [1, 0]
        return torch.tensor(label, dtype=torch.float)

    @property
    def padding_element(self) -> int:
        return 1


@dataclass
class IndexTuple:
    indices: List[int]
    values: List[Any]

    def is_valid(self) -> bool:
        """Return true no indices are duplicate."""
        return len(set(self.indices)) == len(self.indices)

    def append(self, index: int, value: Any) -> None:
        self.indices.append(index)
        self.values.append(value)

    def to_tuple(self) -> Tuple[Any, ...]:
        return tuple(self.values)

    def to_list(self) -> List[Any]:
        return self.values


def build_m_tuples(values: List[Any], m: int) -> List[Tuple[Any, ...]]:
    if m < 1:
        return []

    # Initialize with singletons.
    cur_index_tuples = [
        IndexTuple(indices=[i], values=[v_i]) for i, v_i in enumerate(values)
    ]
    prev_index_tuples = cur_index_tuples

    for _ in range(m - 1):
        cur_index_tuples = []
        for index_tuple in prev_index_tuples:
            for i, v_i in enumerate(values):
                new_index_tuple = copy.deepcopy(index_tuple)
                new_index_tuple.append(i, v_i)
                if new_index_tuple.is_valid():
                    cur_index_tuples.append(new_index_tuple)

        prev_index_tuples = cur_index_tuples

    return [index_tuple.to_tuple() for index_tuple in cur_index_tuples]


def decaying_avg_of_inv_exponentials(
    xs: Sequence[int], alpha: float, sigma: float
) -> float:
    values = np.array(xs)
    terms = sigma * np.exp(-values / sigma)
    factors = alpha ** np.arange(len(terms))
    return np.mean(factors * values).item()


class DesperateStudentPairTask(RegressionTask):
    def __init__(self) -> None:
        super().__init__(label="desperate_student_pair")

    def generate_label(self, x: torch.Tensor) -> torch.Tensor:
        values = x.flatten().tolist()
        if len(values) > 2:
            pair_labels = []
            for i, v_i in enumerate(values):
                for j, v_j in enumerate(values):
                    if i != j:
                        pair_labels.append(
                            decaying_avg_of_inv_exponentials(
                                [v_i, v_j], alpha=0.99, sigma=100
                            )
                        )
            label = np.array(pair_labels).mean()
        else:
            label = decaying_avg_of_inv_exponentials(
                values, alpha=0.99, sigma=100
            )
        return torch.tensor(label, dtype=torch.float)

    @property
    def padding_element(self) -> int:
        return 0


class DesperateStudentTripleTask(RegressionTask):
    def __init__(self) -> None:
        super().__init__(label="desperate_student_triple")

    def generate_label(self, x: torch.Tensor) -> torch.Tensor:
        values = x.flatten().tolist()
        if len(values) > 2:
            triple_labels = []
            for i, v_i in enumerate(values):
                for j, v_j in enumerate(values):
                    for k, v_k in enumerate(values):
                        if i != j and j != k and i != k:
                            triple_labels.append(
                                decaying_avg_of_inv_exponentials(
                                    [v_i, v_j, v_k], alpha=0.99, sigma=100
                                )
                            )
            label = np.array(triple_labels).mean()
        else:
            label = decaying_avg_of_inv_exponentials(
                values, alpha=0.99, sigma=100
            )
        return torch.tensor(label, dtype=torch.float)

    @property
    def padding_element(self) -> int:
        return 0


def get_task(task_config: TaskConfig) -> Task:
    n_classes = len(
        np.arange(task_config.min_value, task_config.max_value + 1)
    )

    tasks = {
        "sum": SumTask(),
        "mean": MeanTask(),
        "mode": ModeTask(n_classes),
        "max": MaxTask(),
        "cardinality": CardinalityTask(),
        "longest_seq_length": LongestSeqLenTask(),
        "largest_contiguous_sum": LargestContiguousSumTask(),
        "largest_pair_sum": LargestNTupleSumTask(n=2),
        "largest_triple_sum": LargestNTupleSumTask(n=3),
        "average_pair_sum": AverageMTupleSumTask(m=2),
        "average_triple_sum": AverageMTupleSumTask(m=3),
        "average_4_tuple_sum": AverageMTupleSumTask(m=4),
        "average_5_tuple_sum": AverageMTupleSumTask(m=5),
        "average_10_tuple_sum": AverageMTupleSumTask(m=10),
        "contains_even": ContainsEvenTask(),
        "desperate_student_pair": DesperateStudentPairTask(),
        "desperate_student_triple": DesperateStudentTripleTask(),
    }

    return tasks[task_config.label]


def is_classification_task(task_config: TaskConfig) -> bool:
    return isinstance(get_task(task_config), ClassificationTask)
