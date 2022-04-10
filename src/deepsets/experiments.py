import os
import time
import pandas as pd

import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from .config import Config
from .datasets import SetDataset, get_data_loader
from .networks import count_parameters

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
        self.use_cuda = torch.cuda.is_available()
        self.epoch_counter = 0
        self.config = config

        self.model = model
        if self.use_cuda:
            self.model.cuda()

        self.model_type = f"{config.model.type}"
        if "deepsets" in config.model.type:
            self.model_type += f"_{config.model.accumulator}"

        self.train_set_loader = get_data_loader(
            train_set, config.experiment.batch_size
        )
        self.valid_set_loader = get_data_loader(
            valid_set, config.experiment.batch_size
        )
        self.test_set_loader = get_data_loader(
            test_set, config.experiment.batch_size
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.experiment.lr,
            weight_decay=config.experiment.weight_decay,
        )

        self.loss_fn = LOSS_FNS[config.experiment.loss]

        multisets_id = "multisets" if config.trainset.multisets else "sets"
        model_subdir = os.path.join(
            self.model_type,
            f"{config.trainset.label}-{multisets_id}",
            f"lr:{config.experiment.lr}-"
            + f"wd:{config.experiment.weight_decay}-"
            + f"{config.experiment.batch_size}",
        )
        log_dir = os.path.join(config.paths.log, model_subdir)
        self.summary_writer = SummaryWriter(log_dir=log_dir)

        model_dir = os.path.join(config.paths.models, model_subdir)
        os.makedirs(model_dir, exist_ok=True)
        self.best_model_filename = os.path.join(model_dir, "best_model.pth")

        model_info = {
            "model": [self.model_type],
            "n_params": [count_parameters(self.model)],
        }

        train_set_info = {
            "train_n_samples": [config.trainset.n_samples],
            "train_max_set_size": [config.trainset.max_set_size],
            "train_min_value": [config.trainset.min_value],
            "train_max_value": [config.trainset.max_value],
            "train_label": [config.trainset.label],
            "train_multisets": [config.trainset.multisets],
            "train_label_mean": [train_set.label_mean],
            "train_label_std": [train_set.label_std],
            "train_label_min": [train_set.label_min],
            "train_label_max": [train_set.label_max],
            "train_label_median": [train_set.label_median],
            "train_label_mode": [train_set.label_mode],
            "train_delta": [train_set.delta],
        }

        valid_set_info = {
            "valid_n_samples": [config.validset.n_samples],
            "valid_max_set_size": [config.validset.max_set_size],
            "valid_min_value": [config.validset.min_value],
            "valid_max_value": [config.validset.max_value],
            "valid_label": [config.validset.label],
            "valid_multisets": [config.validset.multisets],
            "valid_label_mean": [valid_set.label_mean],
            "valid_label_std": [valid_set.label_std],
            "valid_label_min": [valid_set.label_min],
            "valid_label_max": [valid_set.label_max],
            "valid_label_median": [valid_set.label_median],
            "valid_label_mode": [valid_set.label_mode],
            "valid_delta": [valid_set.delta],
        }

        test_set_info = {
            "test_n_samples": [config.testset.n_samples],
            "test_max_set_size": [config.testset.max_set_size],
            "test_min_value": [config.testset.min_value],
            "test_max_value": [config.testset.max_value],
            "test_label": [config.testset.label],
            "test_multisets": [config.testset.multisets],
            "test_label_mean": [test_set.label_mean],
            "test_label_std": [test_set.label_std],
            "test_label_min": [test_set.label_min],
            "test_label_max": [test_set.label_max],
            "test_label_median": [test_set.label_median],
            "test_label_mode": [test_set.label_mode],
            "test_delta": [test_set.delta],
        }

        experiment_info = {
            "lr": [config.experiment.lr],
            "weight_decay": [config.experiment.weight_decay],
            "batch_size": [config.experiment.batch_size],
            "loss": [config.experiment.loss],
            "max_epochs": [config.experiment.max_epochs],
            "patience": [config.experiment.patience],
            "min_delta": [config.experiment.min_delta],
            "random_seed": [config.experiment.random_seed],
        }

        self.results = {
            **model_info,
            **train_set_info,
            **valid_set_info,
            **test_set_info,
            **experiment_info,
        }

    def run(self) -> None:
        print("Running experiment with parameters:")
        print(f"\tmodel: {self.model_type}")
        print(f"\tlabel: {self.config.trainset.label}")
        print(f"\tloss: {self.config.experiment.loss}")
        print(
            f"\tmultisets: {'yes' if self.config.trainset.multisets else 'no'}"
        )

        start_time = time.time()
        self.train()
        self.test()
        end_time = time.time()

        print(f"\nExperiment runtime: {end_time - start_time:.3f}s")

        print("Saving results...")
        pd.DataFrame.from_dict(self.results).to_csv(
            self.config.paths.results,
            mode="a",
            header=not os.path.isfile(self.config.paths.results),
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
        for batch in tqdm(self.train_set_loader):
            x, label = batch
            train_loss = self._train_step(x, label)
            total_train_loss += train_loss

            step_counter = n_batches * self.epoch_counter + batch_counter
            self.summary_writer.add_scalar(loss_id, train_loss, step_counter)
            batch_counter += 1

        return total_train_loss / n_batches

    def _train_step(self, x: torch.Tensor, label: torch.Tensor) -> float:
        if self.use_cuda:
            x, label = x.cuda(), label.cuda()

        self.optimizer.zero_grad()
        pred = self.model(x)
        # To prevent error warning about mismatching dimensions.
        pred = pred.squeeze(dim=1)
        the_loss = self.loss_fn(pred, label)

        # print(f"x = {x.squeeze()}")
        # print(f"label = {label}")
        # print(f"pred = {pred}")
        # print(f"loss = {the_loss}")

        the_loss.backward()

        # norm_type = 2.0
        # device = "cpu"
        # total_norm = torch.norm(
        #     torch.stack(
        #         [
        #             torch.norm(p.grad.detach(), norm_type).to(device)
        #             for p in self.model.parameters()
        #         ]
        #     ),
        #     norm_type,
        # )
        # print(f"PRE-CLIP total gradient norm: {total_norm}")

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50.0)

        # total_norm = torch.norm(
        #     torch.stack(
        #         [
        #             torch.norm(p.grad.detach(), norm_type).to(device)
        #             for p in self.model.parameters()
        #         ]
        #     ),
        #     norm_type,
        # )
        # print(f"POST-CLIP total gradient norm: {total_norm}")

        self.optimizer.step()

        the_loss_tensor = the_loss.data
        if self.use_cuda:
            the_loss_tensor = the_loss_tensor.cpu()

        the_loss_numpy = the_loss_tensor.numpy().flatten()
        the_loss_float = float(the_loss_numpy[0])

        return the_loss_float

    def _eval_model(self, data_loader: DataLoader, loss_id: str) -> float:
        self.model.eval()
        n_batches = len(data_loader)
        total_eval_loss = 0.0

        with torch.no_grad():
            batch_counter = 0
            for batch in tqdm(data_loader):
                x, label = batch
                eval_loss = self._eval_step(x, label)
                total_eval_loss += eval_loss

                step_counter = n_batches * self.epoch_counter + batch_counter
                self.summary_writer.add_scalar(
                    loss_id, eval_loss, step_counter
                )
                batch_counter += 1

        return total_eval_loss / n_batches

    def _eval_step(self, x: torch.Tensor, label: torch.Tensor) -> float:
        if self.use_cuda:
            x, label = x.cuda(), label.cuda()

        pred = self.model(x)
        # To prevent error warning about mismatching dimensions.
        pred = pred.squeeze(dim=1)
        the_loss = self.loss_fn(pred, label)

        the_loss_tensor = the_loss.data
        if self.use_cuda:
            the_loss_tensor = the_loss_tensor.cpu()

        the_loss_numpy = the_loss_tensor.numpy().flatten()
        the_loss_float = float(the_loss_numpy[0])

        return the_loss_float

    def _has_loss_improved(self, best_loss: float, new_loss: float) -> bool:
        return self.config.experiment.min_delta < best_loss - new_loss
