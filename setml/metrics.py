from functools import partial

from typing import Dict, Callable, List, Tuple, Optional

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    r2_score,
    max_error,
    mean_absolute_percentage_error,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import torch

from .config import Task as TaskConfig
from .datasets import SetDataset
from .tasks import is_classification_task

MetricFn = Callable[[np.ndarray, np.ndarray], np.float64]
CollateFn = Callable[[List[np.ndarray]], np.ndarray]


def print_metrics(metrics: Dict[str, float]) -> None:
    print("=== Metrics ===")
    for name, metric in metrics.items():
        print(f"{name}: {metric}")


def mean_baseline_mae(dataset: SetDataset) -> float:
    _, y = dataset.to_sklearn()
    y_mean = y.mean()
    return float(np.abs(y - y_mean).mean())


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    Return symmetric mean absolute percentage error.

    None that the output is not in the [0, 200] "percentage" range but rather
    in [0, 2].

    See: https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Symmetric mean absolute percentage error.
    """
    abs_error = np.abs(y_true - y_pred)
    abs_true = np.abs(y_true)
    abs_pred = np.abs(y_pred)

    denominator = (abs_true + abs_pred) / 2.0

    # Mask denominator to avoid runtime warnings due to zero divisions.
    masked_denominator = np.where(denominator < 1e-8, 1.0, denominator)

    # If denominator is 0, then we must have that both true label and prediction
    # are 0, so the error is 0. However, not manually adjusting yields NaNs due
    # to zero division.
    terms = np.where(denominator < 1e-8, 0.0, abs_error / masked_denominator)

    return terms.mean()


def mase(
    y_true: np.ndarray, y_pred: np.ndarray, train_baseline_mae: float
) -> np.float64:
    """
    Return mean absolute scaled error.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        train_baseline_mae: Mean absolute error of a baseline prediction (e.g.
            label mean) over training data.

    Returns:
        Mean absolute scaled error.
    """
    error = y_true - y_pred
    scaled_error = error / train_baseline_mae
    return np.abs(scaled_error).mean()


class MetricsEngine:
    __slots__ = ("metrics", "labels", "preds", "collate_predictions")

    def __init__(self, collate_predictions: CollateFn) -> None:
        """
        Initialize metrics engine.

        Args:
            collate_predictions: Function to collate a list of accumulated
                predictions into a single array.
        """
        self.metrics: Dict[str, MetricFn] = {}
        self.labels: List[np.ndarray] = []
        self.preds: List[np.ndarray] = []

        self.collate_predictions = collate_predictions

    def register_metric(self, name: str, fn: MetricFn) -> None:
        self.metrics[name] = fn

    def accumulate_predictions(
        self, labels: torch.Tensor, preds: torch.Tensor
    ) -> None:
        self.labels.append(labels.numpy())
        self.preds.append(preds.numpy())

    def compute_metrics(self) -> Dict[str, float]:
        labels = self.collate_predictions(self.labels)
        preds = self.collate_predictions(self.preds)

        results = {
            name: float(metric(labels, preds))
            for name, metric in self.metrics.items()
        }

        return results

    def reset(self) -> None:
        self.labels = []
        self.preds = []


def collate_regression_predictions(preds: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(preds)


def collate_classification_predictions(preds: List[np.ndarray]) -> np.ndarray:
    return np.argmax(np.concatenate(preds), axis=1)


def build_metrics_engine(
    task_config: TaskConfig, train_baseline_mae: Optional[float] = None
) -> MetricsEngine:
    engine = None
    if is_classification_task(task_config):
        engine = MetricsEngine(
            collate_predictions=collate_classification_predictions
        )

        engine.register_metric("accuracy", accuracy_score)
        engine.register_metric("balanced_accuracy", balanced_accuracy_score)
        engine.register_metric(
            "f1_weighted", partial(f1_score, average="weighted")
        )
        engine.register_metric(
            "precision", partial(precision_score, average="weighted")
        )
        engine.register_metric(
            "recall", partial(recall_score, average="weighted")
        )
    else:
        if train_baseline_mae is None:
            raise ValueError(
                "For regression metrics, train_baseline_mae must be provided!"
            )

        engine = MetricsEngine(
            collate_predictions=collate_regression_predictions
        )

        engine.register_metric("mse", mean_squared_error)
        engine.register_metric("mae", mean_absolute_error)
        engine.register_metric("med_ae", median_absolute_error)
        engine.register_metric("max_error", max_error)
        engine.register_metric("mape", mean_absolute_percentage_error)
        engine.register_metric("r2", r2_score)
        engine.register_metric("smape", smape)
        engine.register_metric(
            "mase", partial(mase, train_baseline_mae=train_baseline_mae)
        )
    return engine
