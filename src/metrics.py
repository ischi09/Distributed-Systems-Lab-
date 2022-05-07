from functools import partial

from typing import Dict, Callable, List, Tuple

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch

from config import Task as TaskConfig
from tasks import is_classification_task

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def print_metrics(metrics: Dict[str, float]) -> None:
    print("=== Metrics ===")
    for name, metric in metrics.items():
        print(f"{name}: {metric}")


class MetricsEngine:
    __slots__ = "metrics", "labels", "preds"

    def __init__(self) -> None:
        self.metrics: Dict[str, MetricFn] = {}
        self.labels: List[np.ndarray] = []
        self.preds: List[np.ndarray] = []

    def register_metric(self, name: str, fn: MetricFn) -> None:
        self.metrics[name] = fn

    def accumulate_predictions(
        self, labels: torch.Tensor, preds: torch.Tensor
    ) -> None:
        self.labels.append(labels.numpy())
        self.preds.append(preds.numpy())

    def _collate_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def compute_metrics(self) -> Dict[str, float]:
        labels, preds = self._collate_predictions()

        results = {
            name: float(metric(labels, preds))
            for name, metric in self.metrics.items()
        }

        return results

    def reset(self) -> None:
        self.labels = []
        self.preds = []


class RegressionMetricsEngine(MetricsEngine):
    def __init__(self) -> None:
        super().__init__()
        self.register_metric("r2", r2_score)
        self.register_metric("mae", mean_absolute_error)
        self.register_metric("mse", mean_squared_error)

    def _collate_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        labels = np.concatenate(self.labels)
        preds = np.concatenate(self.preds)
        return labels, preds


class ClassificationMetricsEngine(MetricsEngine):
    def __init__(self) -> None:
        super().__init__()
        self.register_metric("accuracy", accuracy_score)
        self.register_metric("balanced_accuracy", balanced_accuracy_score)
        self.register_metric(
            "f1_weighted", partial(f1_score, average="weighted")
        )
        self.register_metric("precision", precision_score)
        self.register_metric("recall", recall_score)
        self.register_metric(
            "roc_auc_weighted", partial(roc_auc_score, average="weighted")
        )

    def _collate_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        labels = np.argmax(np.concatenate(self.labels), axis=1)
        preds = np.argmax(np.concatenate(self.preds), axis=1)
        return labels, preds


def build_metrics_engine(task_config: TaskConfig) -> MetricsEngine:
    engine = None
    if is_classification_task(task_config):
        engine = ClassificationMetricsEngine()
    else:
        engine = RegressionMetricsEngine()
    return engine
