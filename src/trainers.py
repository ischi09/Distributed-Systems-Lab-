import os

from typing import Dict, Tuple

from tqdm import tqdm

import torch
from torch import optim
from torch.utils.data import DataLoader

from repset.repset.models import RepSet

from config import Config
from datasets import SetDataset, get_data_loader
from tasks import get_task, is_classification_task
from models import build_model
from metrics import build_metrics_engine, print_metrics
from log import build_summary_writer, get_model_subdir


class Trainer:
    def __init__(self) -> None:
        pass

    def train(
        self, train_set: SetDataset, valid_set: SetDataset
    ) -> Dict[str, float]:
        raise NotImplementedError

    def test(self, test_set: SetDataset) -> Dict[str, float]:
        raise NotImplementedError


class TorchTrainer(Trainer):
    def __init__(
        self,
        config: Config,
        model: torch.nn.Module,
    ) -> None:
        super().__init__()

        self.device = torch.device(
            "cuda" if config.experiment.use_gpu else "cpu"
        )

        self.config = config

        self.model = model.to(self.device)
        model_dir = os.path.join(
            config.paths.checkpoints, get_model_subdir(config)
        )
        os.makedirs(model_dir, exist_ok=True)
        self.best_model_filename = os.path.join(model_dir, "best_model.pth")

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.experiment.lr,
            weight_decay=config.experiment.weight_decay,
        )

        task = get_task(config.task)
        self.loss_fn = task.loss_fn
        self.is_classification_task = is_classification_task(config.task)

        self.metrics_engine = build_metrics_engine(config.task)

        self.summary_writer = build_summary_writer(config)

    def train(
        self, train_set: SetDataset, valid_set: SetDataset
    ) -> Dict[str, float]:
        train_set_loader = get_data_loader(
            dataset=train_set,
            batch_size=self.config.experiment.batch_size,
            use_batch_sampler=self.config.experiment.use_batch_sampler,
        )
        valid_set_loader = get_data_loader(
            dataset=valid_set,
            batch_size=self.config.experiment.batch_size,
            use_batch_sampler=self.config.experiment.use_batch_sampler,
        )
        train_class_weights = train_set.class_weights.to(self.device)

        epoch_counter = 0

        avg_train_loss = float("inf")
        best_valid_loss = float("inf")
        n_no_improvement_epochs = 0
        for _ in range(self.config.experiment.max_epochs):
            print(f"\n*** Epoch {epoch_counter} ***")

            print("Training model...")
            avg_train_loss, train_metrics = self._train_model(
                data_loader=train_set_loader,
                class_weights=train_class_weights,
                loss_id="train_loss",
                epoch_counter=epoch_counter,
            )
            print(f"Average train loss: {avg_train_loss}")
            print_metrics(train_metrics)

            print("\nValidating model...")
            avg_valid_loss, valid_metrics = self._eval_model(
                data_loader=valid_set_loader,
                loss_id="valid_loss",
                epoch_counter=epoch_counter,
            )
            print(f"Average validation loss: {avg_valid_loss}")
            print_metrics(valid_metrics)

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

            epoch_counter += 1

        return {
            "epochs": epoch_counter,
            "avg_train_loss": avg_train_loss,
            "best_valid_loss": best_valid_loss,
        }

    def test(self, test_set: SetDataset) -> Dict[str, float]:
        test_set_loader = get_data_loader(
            dataset=test_set,
            batch_size=self.config.experiment.batch_size,
            use_batch_sampler=self.config.experiment.use_batch_sampler,
        )

        print("\nTesting model...")
        self.model.load_state_dict(torch.load(self.best_model_filename))
        avg_test_loss, test_metrics = self._eval_model(
            data_loader=test_set_loader,
            loss_id="test_loss",
            epoch_counter=0,
        )
        print(f"Average test loss: {avg_test_loss}")
        print_metrics(test_metrics)

        test_metrics = {
            f"test_{name}": metric for name, metric in test_metrics.items()
        }
        test_metrics.update({"avg_test_loss": avg_test_loss})
        return test_metrics

    def _train_model(
        self,
        data_loader: DataLoader,
        class_weights: torch.Tensor,
        loss_id: str,
        epoch_counter: int,
    ) -> Tuple[float, Dict[str, float]]:
        self.metrics_engine.reset()

        self.model.train()
        n_batches = len(data_loader)
        total_train_loss = 0.0

        batch_counter = 0
        n_samples = 0
        for batch in tqdm(data_loader):
            x, mask, label = batch
            x = x.to(self.device)
            mask = mask.to(self.device)
            label = label.to(self.device)

            train_loss = self._train_step(
                x, mask, label, class_weights=class_weights
            )
            n_samples += len(batch)
            total_train_loss += train_loss * len(batch)

            step_counter = n_batches * epoch_counter + batch_counter
            self.summary_writer.add_scalar(loss_id, train_loss, step_counter)
            batch_counter += 1

        return (
            total_train_loss / n_samples,
            self.metrics_engine.compute_metrics(),
        )

    def _train_step(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        label: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> float:
        self.optimizer.zero_grad()
        pred = self.model(x, mask)
        # To prevent error warning about mismatching dimensions.
        pred = pred.squeeze(dim=1)

        self.metrics_engine.accumulate_predictions(
            label.detach().cpu(), pred.detach().cpu()
        )

        if self.is_classification_task:
            the_loss = self.loss_fn(pred, label, weight=class_weights)
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

    def _eval_model(
        self, data_loader: DataLoader, loss_id: str, epoch_counter: int
    ) -> Tuple[float, Dict[str, float]]:
        self.metrics_engine.reset()

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

                step_counter = n_batches * epoch_counter + batch_counter
                self.summary_writer.add_scalar(
                    loss_id, eval_loss, step_counter
                )
                batch_counter += 1

        return (
            total_eval_loss / n_samples,
            self.metrics_engine.compute_metrics(),
        )

    def _eval_step(
        self, x: torch.Tensor, mask: torch.Tensor, label: torch.Tensor
    ) -> float:
        pred = self.model(x, mask)
        # To prevent error warning about mismatching dimensions.
        pred = pred.squeeze(dim=1)
        the_loss = self.loss_fn(pred, label)

        self.metrics_engine.accumulate_predictions(
            label.detach().cpu(), pred.detach().cpu()
        )

        the_loss_tensor = the_loss.detach().cpu().data
        the_loss_numpy = the_loss_tensor.numpy().flatten()
        the_loss_float = float(the_loss_numpy[0])

        return the_loss_float

    def _has_loss_improved(self, best_loss: float, new_loss: float) -> bool:
        return self.config.experiment.min_delta < best_loss - new_loss


class RepSetTrainer(Trainer):
    def __init__(self, config: Config, model: RepSet) -> None:
        super().__init__()

        self.config = config
        self.model = model

        self.metrics_engine = build_metrics_engine(config.task)

    def train(
        self, train_set: SetDataset, valid_set: SetDataset
    ) -> Dict[str, float]:
        train_set_loader = get_data_loader(
            dataset=train_set,
            batch_size=self.config.experiment.batch_size,
            use_batch_sampler=self.config.experiment.use_batch_sampler,
        )
        valid_set_loader = get_data_loader(
            dataset=valid_set,
            batch_size=self.config.experiment.batch_size,
            use_batch_sampler=self.config.experiment.use_batch_sampler,
        )

        epoch_counter = 0

        avg_train_loss = float("inf")
        best_valid_loss = float("inf")
        n_no_improvement_epochs = 0
        for _ in range(self.config.experiment.max_epochs):
            print(f"\n*** Epoch {epoch_counter} ***")

            print("Training model...")
            avg_train_loss, train_metrics = self._train_model(
                data_loader=train_set_loader,
            )
            print(f"Average train loss: {avg_train_loss}")
            print_metrics(train_metrics)

            print("\nValidating model...")
            avg_valid_loss, valid_metrics = self._eval_model(
                data_loader=valid_set_loader,
                loss_id="valid_loss",
                epoch_counter=epoch_counter,
            )
            print(f"Average validation loss: {avg_valid_loss}")
            print_metrics(valid_metrics)

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

            epoch_counter += 1

        return {
            "epochs": epoch_counter,
            "avg_train_loss": avg_train_loss,
            "best_valid_loss": best_valid_loss,
        }

    def test(self, test_set: SetDataset) -> Dict[str, float]:
        return super().test(test_set)

    def _train_model(
        self,
        data_loader: DataLoader,
    ) -> Tuple[float, Dict[str, float]]:
        self.metrics_engine.reset()

        total_train_loss = 0.0

        n_samples = 0
        for batch in tqdm(data_loader):
            x, _, label = batch

            pred = self.model.train(X=x.numpy(), y=label.numpy())

            # TODO: computer train loss

            self.metrics_engine.accumulate_predictions(
                label.detach().cpu(), torch.tensor(pred)
            )

        return (
            total_train_loss / n_samples,
            self.metrics_engine.compute_metrics(),
        )


def build_trainer(config: Config, delta: float) -> Trainer:
    model = build_model(
        config=config,
        delta=delta,
    )
    if config.model.type == "rep_set":
        return RepSetTrainer(config=config, model=model)
    return TorchTrainer(config=config, model=model)
