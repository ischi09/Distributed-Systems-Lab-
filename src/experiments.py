import os
import time

from typing import Any, List, Dict

import pandas as pd

from config import Config
from datasets import SetDataset
from tasks import get_task
from models import count_parameters
from trainers import Trainer


def to_list_dict(d: Dict[str, Any]) -> Dict[str, List[Any]]:
    return {k: [v] for k, v in d.items()}


def dataset_stats_dict(dataset: SetDataset, kind: str) -> Dict[str, Any]:
    return {
        f"{kind}_label_mean": dataset.label_mean,
        f"{kind}_label_std": dataset.label_std,
        f"{kind}_label_min": dataset.label_min,
        f"{kind}_label_max": dataset.label_max,
        f"{kind}_label_median": dataset.label_median,
        f"{kind}_label_mode": dataset.label_mode,
        f"{kind}_label_entropy": dataset.label_entropy,
        f"{kind}_delta": dataset.delta,
    }


class Experiment:
    def __init__(
        self,
        config: Config,
        train_set: SetDataset,
        valid_set: SetDataset,
        test_set: SetDataset,
        trainer: Trainer,
    ):
        self.config = config

        self.trainer = trainer
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set

        self._init_results_logging()

    def _init_results_logging(self) -> None:
        model_info = self.config.model._content
        model_info["n_params"] = count_parameters(self.trainer.model)

        task_info = self.config.task._content

        datasets_info = self.config.datasets._content

        train_set_info = dataset_stats_dict(self.train_set, "train")
        valid_set_info = dataset_stats_dict(self.valid_set, "valid")
        test_set_info = dataset_stats_dict(self.test_set, "test")

        experiment_info = self.config.experiment._content
        task = get_task(self.config.task)
        experiment_info["loss"] = task.loss

        print("*** Experiment Setup ***")

        def print_info(heading: str, d: Dict[str, Any]) -> None:
            msg = f"\n{heading}:\n"
            for k, v in d.items():
                msg += " " * 4 + f"{k}: {v}\n"
            print(msg)

        print_info("Model", model_info)
        print_info("Task", task_info)
        print_info("Datasets", datasets_info)
        print_info("Trainset Statistics", train_set_info)
        print_info("Validset Statistics", valid_set_info)
        print_info("Testset Statistics", test_set_info)
        print_info("Experiment", experiment_info)

        self.results = {
            **to_list_dict(model_info),
            **to_list_dict(task_info),
            **to_list_dict(datasets_info),
            **to_list_dict(train_set_info),
            **to_list_dict(valid_set_info),
            **to_list_dict(test_set_info),
            **to_list_dict(experiment_info),
        }

    def run(self) -> None:
        start_time = time.time()
        train_results = self.trainer.train(
            train_set=self.train_set, valid_set=self.valid_set
        )
        test_results = self.trainer.test(test_set=self.test_set)
        end_time = time.time()

        self.results.update(to_list_dict(train_results))
        self.results.update(to_list_dict(test_results))

        print(f"\nExperiment runtime: {end_time - start_time:.3f}s")

        print("Saving results...")
        os.makedirs(self.config.paths.results, exist_ok=True)
        results_filename = os.path.join(
            self.config.paths.results, self.config.experiment.results_out
        )
        pd.DataFrame.from_dict(self.results).to_csv(
            results_filename,
            mode="a",
            header=not os.path.isfile(results_filename),
            index=False,
        )
        print("Done!")
