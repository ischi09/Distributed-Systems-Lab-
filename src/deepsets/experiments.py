import os
import pandas as pd

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from .config import Config
from .datasets import SetDataset
from .networks import count_parameters

LOSS_FNS = {"mse": F.mse_loss, "ce": F.cross_entropy}


class Experiment:
    def __init__(
        self,
        config: Config,
        model: torch.nn.Module,
        model_name: str,  # TODO: not elegant, maybe experiment name in config better
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

        self.model_name = model_name

        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set

        lr = config.experiment.lr
        weight_decay = config.experiment.weight_decay

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.loss_fn = LOSS_FNS[config.experiment.loss]

        self.summary_writer = SummaryWriter(
            log_dir=f"{config.paths.log}/exp-lr:{lr}-wd:{weight_decay}"
        )

        self.results = pd.DataFrame(
            columns=[
                "model",
                "n_params",
                "n_samples",
                "max_set_size",
                "min_value",
                "max_value",
                "label",
                "multisets",
                "loss",
                "epochs",
                "lr",
                "weight_decay",
                "avg_test_loss",
            ]
        )

    def run(self) -> None:
        print("Running experiment with parameters:")
        print(f"    label: {self.config.trainset.label}")
        print(f"    loss: {self.config.experiment.loss}")
        print(f"    multisets: {'yes' if self.config.trainset.multisets else 'no'}")

        self.train()
        self.test()

        print("Saving results...")
        self.results.to_csv(
            self.config.paths.results,
            mode="a",
            header=not os.path.isfile(self.config.paths.results),
            index=False,
        )

    def train(self) -> None:
        for _ in range(self.config.experiment.epochs):
            print(f"\n*** Epoch {self.epoch_counter} ***")

            print("Training model...")
            avg_train_loss = self.__train_model("train_loss")
            print(f"Average train loss: {avg_train_loss}")

            print("Validating model...")
            avg_valid_loss = self.__eval_model(self.valid_set, "valid_loss")
            print(f"Average validation loss: {avg_valid_loss}")

            self.epoch_counter += 1

    def test(self) -> None:
        print("\nTesting model...")
        avg_test_loss = self.__eval_model(self.test_set, "test_loss")
        print(f"Average test loss: {avg_test_loss}")

        testset_config = self.config.testset
        exp_config = self.config.experiment

        results = {
            "model": self.model_name,  # TODO: Generate from model.
            "n_params": count_parameters(self.model),
            "n_samples": testset_config.n_samples,
            "max_set_size": testset_config.max_set_size,
            "min_value": testset_config.min_value,
            "max_value": testset_config.max_value,
            "label": testset_config.label,
            "multisets": testset_config.multisets,
            "loss": exp_config.loss,
            "epochs": exp_config.epochs,
            "lr": exp_config.lr,
            "weight_decay": exp_config.weight_decay,
            "avg_test_loss": avg_test_loss,
        }
        self.results.loc[len(self.results.index)] = list(results.values())

    def __train_model(self, loss_id: str) -> None:
        self.model.train()
        train_set_size = len(self.train_set)
        total_train_loss = 0.0

        for i in tqdm(range(train_set_size)):
            x, label = self.train_set[i]
            train_loss = self.__train_step(x, label)
            total_train_loss += train_loss

            step_counter = train_set_size * self.epoch_counter + i
            self.summary_writer.add_scalar(loss_id, train_loss, step_counter)

        return total_train_loss / train_set_size

    def __train_step(self, x: torch.FloatTensor, label: torch.FloatTensor) -> float:
        if self.use_cuda:
            x, label = x.cuda(), label.cuda()

        self.optimizer.zero_grad()
        pred = self.model(x)
        # To prevent error warning about mismatching dimensions.
        pred = torch.squeeze(pred, dim=0)
        the_loss = self.loss_fn(pred, label)

        the_loss.backward()
        self.optimizer.step()

        the_loss_tensor = the_loss.data
        if self.use_cuda:
            the_loss_tensor = the_loss_tensor.cpu()

        the_loss_numpy = the_loss_tensor.numpy().flatten()
        the_loss_float = float(the_loss_numpy[0])

        return the_loss_float

    def __eval_model(self, dataset: SetDataset, loss_id: str) -> float:
        self.model.eval()
        dataset_size = len(dataset)
        total_eval_loss = 0.0

        with torch.no_grad():
            for i in tqdm(range(dataset_size)):
                x, label = dataset[i]
                eval_loss = self.__eval_step(x, label)
                total_eval_loss += eval_loss

                step_counter = dataset_size * self.epoch_counter + i
                self.summary_writer.add_scalar(loss_id, eval_loss, step_counter)

        return total_eval_loss / dataset_size

    def __eval_step(self, x: torch.FloatTensor, label: torch.FloatTensor) -> float:
        if self.use_cuda:
            x, label = x.cuda(), label.cuda()

        pred = self.model(x)
        # To prevent error warning about mismatching dimensions.
        pred = torch.squeeze(pred, dim=0)
        the_loss = self.loss_fn(pred, label)

        the_loss_tensor = the_loss.data
        if self.use_cuda:
            the_loss_tensor = the_loss_tensor.cpu()

        the_loss_numpy = the_loss_tensor.numpy().flatten()
        the_loss_float = float(the_loss_numpy[0])

        return the_loss_float
