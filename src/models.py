from math import log, floor

from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn

from config import Config
from tasks import ClassificationTask, get_task
from set_transformer.modules import SAB, PMA
from fspool.fspool import FSPool
from repset.repset.models import RepSet


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
        x = torch.permute(x, (0, 2, 1))
        x, _ = self.pool(x, n=mask.sum(dim=1).squeeze(dim=-1))
        return self.rho(x)


def build_model(config: Config, delta: float) -> nn.Module:
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
    elif model_config.type == "rep_set":
        model = RepSet(
            lr=config.experiment.lr,
            n_hidden_sets=10,
            n_elements=config.task.max_set_size,
            d=config.model.data_dim,
            n_classes=task.n_classes,
        )
    return model
