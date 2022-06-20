from typing import Callable, Union

import sklearn
from sklearn.dummy import DummyRegressor
import torch
import torch.nn as nn

from .config import Config
from .tasks import ClassificationTask, get_task
from .set_transformer.modules import SAB, PMA
from .fspool.fspool import FSPool


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def masked_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x = x * mask
    return x.sum(axis=1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sum = masked_sum(x, mask)
    n_elements = mask.sum(axis=1)
    return sum / n_elements


def masked_std(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = masked_mean(x, mask).unsqueeze(dim=-1)
    variance = masked_mean(torch.square(x - mean), mask)
    return torch.sqrt(variance)


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg_infinity = torch.tensor(float("-inf")).to(x.device)
    x = torch.where(mask.bool(), x, neg_infinity)
    return x.max(dim=1).values


def masked_min(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return -masked_max(-x, mask)


ACCUMLATORS = {
    "sum": masked_sum,
    "mean": masked_mean,
    "std": masked_std,
    "max": masked_max,
    "min": masked_min,
}

MaskedAccumulator = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class DeepSetsInvariant(nn.Module):
    def __init__(
        self,
        phi: nn.Module,
        rho: nn.Module,
        accumulator: MaskedAccumulator,
    ):
        super().__init__()
        self.phi = phi
        self.rho = rho
        self.accumulator = accumulator

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.phi(x)  # x.shape = (batch_size, set_size, input_dim)
        x = self.accumulator(x, mask)
        return self.rho(x)


class InternalMlp(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

        self.fully_conn_1 = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=50), nn.PReLU()
        )
        self.fully_conn_2 = nn.Sequential(
            nn.Linear(in_features=50, out_features=50), nn.PReLU()
        )
        self.final = nn.Linear(in_features=50, out_features=output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fully_conn_1(x)
        x = self.fully_conn_2(x)
        x = self.final(x)
        return x


class Mlp(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mlp = InternalMlp(input_dim=input_dim, output_dim=output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(dim=-1)
        return self.mlp(x)


class SortedMlp(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mlp = Mlp(input_dim=input_dim, output_dim=output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x, _ = torch.sort(x, dim=1)
        return self.mlp(x, mask)


class Pna(nn.Module):
    def __init__(self, mlp: nn.Module, delta: float):
        super().__init__()
        self.mlp = mlp
        self.delta = delta

    def scale_amplification(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        scale = torch.log(mask.sum(dim=1) + 1) / self.delta
        return x * scale

    def scale_attenuation(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        scale = self.delta / torch.log(mask.sum(dim=1) + 1)
        return x * scale

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Aggregration
        mean = ACCUMLATORS["mean"](x, mask)
        agg_max = ACCUMLATORS["max"](x, mask)
        agg_min = ACCUMLATORS["min"](x, mask)
        std = ACCUMLATORS["std"](x, mask)

        aggr_concat = torch.cat((mean, agg_max, agg_min, std), dim=1)

        # Scaling
        identity = aggr_concat
        amplification = self.scale_amplification(aggr_concat, mask)
        attenuation = self.scale_attenuation(aggr_concat, mask)

        scale_concat = torch.cat((identity, amplification, attenuation), dim=1)

        return self.mlp(scale_concat)


class SmallSetTransformer(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=1, dim_out=64, num_heads=4),
            SAB(dim_in=64, dim_out=64, num_heads=4),
        )
        self.dec = nn.Sequential(
            PMA(dim=64, num_heads=4, num_seeds=1),
            nn.Linear(in_features=64, out_features=output_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(dim=-1)


class DeepSetsInvariantFSPool(nn.Module):
    def __init__(
        self,
        phi: nn.Module,
        rho: nn.Module,
        pool: FSPool,
    ):
        super().__init__()
        self.phi = phi
        self.rho = rho
        self.pool = pool

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.phi(x)  # x.shape = (batch_size, set_size, input_dim)
        # x.shape = (batch_size, latent_dim, set_size), as FSPool requires
        # set_dim last.
        x = x.permute((0, 2, 1))
        x, _ = self.pool(x, n=mask.sum(dim=1).squeeze(dim=-1))
        return self.rho(x)


class GRUNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        drop_prob=0.2,
    ):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            drop_prob=drop_prob,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.relu = nn.ReLu

    # Not hundred percent sure how i should handle the hidden states here.
    # Will need to pick one of these
    def forward(self, x) -> torch.Tensor:
        """Hidden state generated through Forward propagation"""
        h0 = torch.zeros(
            self.n_layers, x.size(0), self.hidden_dim
        ).requires_grad_()
        out, _ = self.gru(x, h0.detach())

        # Need to reshape the output for the FC layer
        # This output in shape (batch_size, output_dim), probs need reshapeing
        return self.fc(out[:, -1, :])


# Code from another source which seems to handle the hidden layers differently
#     def forward(self, x, h):
#         out, h = self.gru(x, h)
#         out = self.fc(self.relu(out[:,-1]))
#         return out, h

#     def init_hidden(self, batch_size):
#         weight = next(self.parameters()).data
#         hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
#         return hidden


class LSTMNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        drop_prob=0.2,
    ):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=drop_prob,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.relu = nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        """Hidden State generated through Forward propagation again"""
        h0 = torch.zeros(
            self.n_layers, x.size(0), self.hidden_dim
        ).requires_grad_()

        c0 = torch.zeros(
            self.n_layers, x.size(0), self.hidden_dim
        ).requires_grad_()
        # Need to detach, otherwise we go all the way to the front
        out, (_, _) = self.lstm(x, (h0.detach(), c0.detach()))

        # Need to reshape the output for the FC layer
        # This output in shape (batch_size, output_dim), probs need reshapeing
        return self.fc(out[:, -1, :])


# Code from another source which seems to handle the hidden layers differently
#     def forward(self, x, h):
#         out, h = self.lstm(x, h)
#         out = self.fc(self.relu(out[:,-1]))
#         return out, h

#     def init_hidden(self, batch_size):
#         weight = next(self.parameters()).data
#         hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
#                   weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
#         return hidden


class DeepSetsDs1t(nn.Module):
    """
    DeepSets model optimized for Desperate Student 1 Tuple task.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(DeepSetsDs1t, self).__init__()

        self.phi = nn.Sequential(
            nn.Linear(input_dim, 30),
            nn.ELU(inplace=True),
            nn.Linear(30, 50),
            nn.ELU(inplace=True),
            nn.Linear(50, 50),
            nn.ELU(inplace=True),
            nn.Linear(50, hidden_dim),
        )

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, 50),
            nn.ELU(inplace=True),
            nn.Linear(50, 30),
            nn.ELU(inplace=True),
            nn.Linear(30, 10),
            nn.ELU(inplace=True),
            nn.Linear(10, output_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x.shape = [batch_size, set_size, input_dim]
        x = self.phi(x)
        x = masked_sum(x, mask)
        return self.rho(x)


class SortedMlpDs1t(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 30),
            nn.ELU(inplace=True),
            nn.Linear(30, 90),
            nn.ELU(inplace=True),
            nn.Linear(90, 90),
            nn.ELU(inplace=True),
            nn.Linear(90, 30),
            nn.ELU(inplace=True),
            nn.Linear(30, 10),
            nn.ELU(inplace=True),
            nn.Linear(10, output_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x, _ = torch.sort(x, dim=1)
        x = x.squeeze(-1)
        return self.mlp(x)


def build_model(
    config: Config, delta: float
) -> Union[nn.Module, sklearn.base.BaseEstimator]:
    model_config = config.model
    task = get_task(config.task)

    if isinstance(task, ClassificationTask):
        output_dim = task.n_classes
    else:
        output_dim = model_config.data_dim

    model = None
    if "deepsets_mlp" in model_config.type:
        accumulator = model_config.type.split("_")[-1]

        phi = InternalMlp(
            input_dim=model_config.data_dim,
            output_dim=config.task.max_set_size,
        )

        rho = InternalMlp(
            input_dim=config.task.max_set_size,
            output_dim=output_dim,
        )

        if accumulator in ACCUMLATORS.keys():
            model = DeepSetsInvariant(
                phi=phi,
                rho=rho,
                accumulator=ACCUMLATORS[accumulator],
            )
        elif accumulator == "fspool":
            model = DeepSetsInvariantFSPool(
                phi=phi,
                rho=rho,
                pool=FSPool(
                    in_channels=config.task.max_set_size,
                    n_pieces=10,  # TODO: should be a config parameter
                ),
            )
    elif model_config.type == "pna":
        model = Pna(
            mlp=InternalMlp(input_dim=12, output_dim=output_dim),
            delta=delta,
        )
    elif model_config.type == "mlp":
        model = Mlp(
            input_dim=config.task.max_set_size,
            output_dim=output_dim,
        )
    elif model_config.type == "sorted_mlp":
        model = SortedMlp(
            input_dim=config.task.max_set_size,
            output_dim=output_dim,
        )
    elif model_config.type == "small_set_transformer":
        model = SmallSetTransformer(output_dim=output_dim)
    # TODO need to know how to handle the hidden dimension, n_layers and DROP PROBABLITY!! on initailisation.
    elif model_config.type == "lstm":
        model = LSTMNet(
            input_dim=config.task.max_set_size,
            hidden_dim=9,
            output_dim=output_dim,
            n_layers=9,
        )
    elif model_config.type == "gru":
        model = GRUNet(
            input_dim=config.task.max_set_size,
            hidden_dim=9,
            output_dim=output_dim,
            n_layers=9,
        )
    elif model_config.type == "deepsets_ds1t":
        model = DeepSetsDs1t(
            input_dim=config.model.data_dim,
            hidden_dim=config.model.latent_dim,
            output_dim=config.model.data_dim,
        )
    elif model_config.type == "sorted_mlp_ds1t":
        model = SortedMlpDs1t(
            input_dim=config.task.max_set_size,
            output_dim=output_dim,
        )
    elif model_config.type == "mean_baseline":
        model = DummyRegressor()
    return model
