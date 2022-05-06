import os
import time
import pandas as pd
from functools import partial

from typing import Any, List, Dict

import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from config import Config
from tasks import get_task, ClassificationTask
from datasets import SetDataset, get_data_loader
from networks import count_parameters

LOSS_FNS = {"mse": F.mse_loss, "ce": F.cross_entropy}


class Experiment:
    def __init__(
        self,
        config: Config,
        model: torch.nn.Module,
        train_set: SetDataset,
        valid_set: SetDataset,
        test_set: SetDataset,
    ):
        self.device = torch.device(
            "cuda" if config.experiment.use_gpu else "cpu"
        )
        self.epoch_counter = 0
        self.config = config

        self.model = model.to(self.device)

        self.train_set_loader = get_data_loader(
            dataset=train_set,
            batch_size=config.experiment.batch_size,
            use_batch_sampler=config.experiment.use_batch_sampler,
        )
        self.valid_set_loader = get_data_loader(
            dataset=valid_set,
            batch_size=config.experiment.batch_size,
            use_batch_sampler=config.experiment.use_batch_sampler,
        )
        self.test_set_loader = get_data_loader(
            dataset=test_set,
            batch_size=config.experiment.batch_size,
            use_batch_sampler=config.experiment.use_batch_sampler,
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.experiment.lr,
            weight_decay=config.experiment.weight_decay,
        )

        task = get_task(config.task)
        self.is_classification_task = isinstance(task, ClassificationTask)
        self.train_class_weights = train_set.class_weights.to(self.device)
        self.loss_fn = task.loss_fn

        multisets_id = "multisets" if config.task.multisets else "sets"
        model_subdir = os.path.join(
            config.model.type,
            f"{config.task.label}-{multisets_id}",
            f"lr:{config.experiment.lr}-"
            + f"wd:{config.experiment.weight_decay}-"
            + f"{config.experiment.batch_size}",
        )
        log_dir = os.path.join(config.paths.log, model_subdir)
        self.summary_writer = SummaryWriter(log_dir=log_dir)

        model_dir = os.path.join(config.paths.checkpoints, model_subdir)
        os.makedirs(model_dir, exist_ok=True)
        self.best_model_filename = os.path.join(model_dir, "best_model.pth")

        def to_list_dict(d: Dict[str, Any]) -> Dict[str, List[Any]]:
            return {k: [v] for k, v in d.items()}

        model_info = config.model._content
        model_info["n_params"] = count_parameters(self.model)

        task_info = config.task._content

        datasets_info = config.datasets._content

        def dataset_stats_dict(
            dataset: SetDataset, kind: str
        ) -> Dict[str, Any]:
            return {
                f"{kind}_label_mean": dataset.label_mean,
                f"{kind}_label_std": dataset.label_std,
                f"{kind}_label_min": dataset.label_min,
                f"{kind}_label_max": dataset.label_max,
                f"{kind}_label_median": dataset.label_median,
                f"{kind}_label_mode": dataset.label_mode,
                f"{kind}_delta": dataset.delta,
            }

        train_set_info = dataset_stats_dict(train_set, "train")
        valid_set_info = dataset_stats_dict(valid_set, "valid")
        test_set_info = dataset_stats_dict(test_set, "test")

        experiment_info = config.experiment._content
        experiment_info["loss"] = task.loss

        print("*** Experiment Setup ***\n")

        def print_info(heading: str, d: Dict[str, Any]) -> None:
            print(f"{heading}:")
            for k, v in d.items():
                print(" " * 4 + f"{k}: {v}")
            print("")

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
        self.train()
        self.test()
        end_time = time.time()

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

    def train(self) -> None:
        avg_train_loss = float("inf")
        best_valid_loss = float("inf")
        n_no_improvement_epochs = 0
        for _ in range(self.config.experiment.max_epochs):
            print(f"\n*** Epoch {self.epoch_counter} ***")

            print("Training model...")
            avg_train_loss = self._train_model("train_loss")
            print(f"Average train loss: {avg_train_loss}")

            print("Validating model...")
            avg_valid_loss = self._eval_model(
                self.valid_set_loader, "valid_loss"
            )
            print(f"Average validation loss: {avg_valid_loss}")

            if self._has_loss_improved(best_valid_loss, avg_valid_loss):
                loss_improvement = abs(best_valid_loss - avg_valid_loss)
                print(
                    f"Best validation loss has improved by {loss_improvement}!"
                )

                torch.save(self.model.state_dict(), self.best_model_filename)
                best_valid_loss = avg_valid_loss
                n_no_improvement_epochs = 0
            else:
                n_no_improvement_epochs += 1

            if n_no_improvement_epochs >= self.config.experiment.patience:
                break

            self.epoch_counter += 1

        self.results["epochs"] = [self.epoch_counter]
        self.results["avg_train_loss"] = [avg_train_loss]
        self.results["best_valid_loss"] = [best_valid_loss]

    def test(self) -> None:
        print("\nTesting model...")
        self.model.load_state_dict(torch.load(self.best_model_filename))
        avg_test_loss = self._eval_model(self.test_set_loader, "test_loss")
        print(f"Average test loss: {avg_test_loss}")

        self.results["avg_test_loss"] = [avg_test_loss]

    def _train_model(self, loss_id: str) -> float:
        self.model.train()
        n_batches = len(self.train_set_loader)
        total_train_loss = 0.0

        batch_counter = 0
        n_samples = 0
        for batch in tqdm(self.train_set_loader):
            x, mask, label = batch
            x = x.to(self.device)
            mask = mask.to(self.device)
            label = label.to(self.device)

            train_loss = self._train_step(x, mask, label)
            n_samples += len(batch)
            total_train_loss += train_loss * len(batch)

            step_counter = n_batches * self.epoch_counter + batch_counter
            self.summary_writer.add_scalar(loss_id, train_loss, step_counter)
            batch_counter += 1

        return total_train_loss / n_samples

    def _train_step(
        self, x: torch.Tensor, mask: torch.Tensor, label: torch.Tensor
    ) -> float:
        self.optimizer.zero_grad()
        pred = self.model(x, mask)
        # To prevent error warning about mismatching dimensions.
        pred = pred.squeeze(dim=1)

        if self.is_classification_task:
            the_loss = self.loss_fn(
                pred, label, weight=self.train_class_weights
            )
        else:
            the_loss = self.loss_fn(pred, label)

        the_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.experiment.grad_norm_threshold
        )

        self.optimizer.step()

        the_loss_tensor = the_loss.detach().cpu().data
        the_loss_numpy = the_loss_tensor.numpy().flatten()
        the_loss_float = float(the_loss_numpy[0])

        return the_loss_float

    def _eval_model(self, data_loader: DataLoader, loss_id: str) -> float:
        self.model.eval()
        n_batches = len(data_loader)
        total_eval_loss = 0.0

        with torch.no_grad():
            batch_counter = 0
            n_samples = 0
            for batch in tqdm(data_loader):
                x, mask, label = batch
                x = x.to(self.device)
                mask = mask.to(self.device)
                label = label.to(self.device)

                eval_loss = self._eval_step(x, mask, label)
                n_samples += len(batch)
                total_eval_loss += eval_loss * len(batch)

                step_counter = n_batches * self.epoch_counter + batch_counter
                self.summary_writer.add_scalar(
                    loss_id, eval_loss, step_counter
                )
                batch_counter += 1

        return total_eval_loss / n_samples

    def _eval_step(
        self, x: torch.Tensor, mask: torch.Tensor, label: torch.Tensor
    ) -> float:
        pred = self.model(x, mask)
        # To prevent error warning about mismatching dimensions.
        pred = pred.squeeze(dim=1)
        the_loss = self.loss_fn(pred, label)

        the_loss_tensor = the_loss.detach().cpu().data
        the_loss_numpy = the_loss_tensor.numpy().flatten()
        the_loss_float = float(the_loss_numpy[0])

        return the_loss_float

    def _has_loss_improved(self, best_loss: float, new_loss: float) -> bool:
        return self.config.experiment.min_delta < best_loss - new_loss
