from typing import Callable

import math
import torch
import torch.nn as nn

from .config import Model as ModelConfig
from .config import Dataset as DatasetConfig
from .tasks import ClassificationTask, get_task
from .set_transformer.modules import SAB, PMA
from .fspool.fspool import FSPool


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


ACCUMLATORS = {
    "sum": lambda x: x.sum(dim=1),
    "mean": lambda x: x.mean(dim=1),
    "std": lambda x: x.std(dim=1, unbiased=False),
    "max": lambda x: x.max(dim=1).values,
    "min": lambda x: x.min(dim=1).values,
}


def generate_model(
    model_config: ModelConfig, dataset_config: DatasetConfig, delta: float
) -> nn.Module:
    task = get_task(dataset_config)

    if isinstance(task, ClassificationTask):
        output_dim = task.n_classes
    else:
        output_dim = model_config.data_dim

    model = None
    if model_config.type == "deepsets_mlp":
        model = DeepSetsInvariant(
            phi=MLP(
                input_dim=model_config.data_dim,
                hidden_dim=10,
                output_dim=model_config.laten_dim,
            ),
            rho=MLP(
                input_dim=model_config.laten_dim,
                hidden_dim=10,
                output_dim=output_dim,
            ),
            accumulator=ACCUMLATORS[model_config.accumulator],
        )
    elif model_config.type == "pna":
        model = PNA(
            mlp=MLP(input_dim=12, hidden_dim=10, output_dim=output_dim),
            delta=delta,
        )
    elif model_config.type == "sorted_mlp":
        model = SortedMLP(
            input_dim=10,
            hidden_dim=10,
            output_dim=output_dim,
        )
    elif model_config.type == "small_set_transformer":
        model = SmallSetTransformer(output_dim=output_dim)
    elif model_config.type == "deepsets_mlp_fspool":
        model = DeepSetsInvariantFSPool(
            phi=MLP(
                input_dim=model_config.data_dim,
                hidden_dim=10,
                output_dim=model_config.laten_dim,
            ),
            rho=MLP(
                input_dim=model_config.laten_dim,
                hidden_dim=10,
                output_dim=output_dim,
            ),
            pool=FSPool(
                in_channels=model_config.laten_dim,
                n_pieces=10,  # TODO: should be a config parameter
            ),
        )
    return model


class DeepSetsInvariant(nn.Module):
    def __init__(
        self,
        phi: nn.Module,
        rho: nn.Module,
        accumulator: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.phi = phi
        self.rho = rho
        self.accumulator = accumulator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.phi(x)  # x.shape = (batch_size, set_size, input_dim)
        x = self.accumulator(x)
        return self.rho(x)


class PNA(nn.Module):
    def __init__(self, mlp: nn.Module, delta: float):
        super().__init__()
        self.mlp = mlp
        self.delta = delta

    def scale_amplification(
        self, x: torch.Tensor, degree: int
    ) -> torch.Tensor:
        scale = math.log(degree + 1) / self.delta
        return x * scale

    def scale_attenuation(self, x: torch.Tensor, degree: int) -> torch.Tensor:
        scale = self.delta / math.log(degree + 1)
        return x * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Aggregration
        mean = ACCUMLATORS["mean"](x)
        agg_max = ACCUMLATORS["max"](x)
        agg_min = ACCUMLATORS["min"](x)
        std = ACCUMLATORS["std"](x)

        aggr_concat = torch.cat((mean, agg_max, agg_min, std), dim=1)

        # Scaling
        identity = aggr_concat
        amplification = self.scale_amplification(
            aggr_concat, degree=x.size(dim=1)
        )
        attenuation = self.scale_attenuation(aggr_concat, degree=x.size(dim=1))

        scale_concat = torch.cat((identity, amplification, attenuation), dim=1)

        return self.mlp(scale_concat)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SortedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = torch.sort(x, dim=1)
        x = x.squeeze(dim=-1)
        return self.layers(x)


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

    def forward(self, x):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.phi(x)  # x.shape = (batch_size, set_size, input_dim)
        # x.shape = (batch_size, latent_dim, set_size), as FSPool requires
        # set_dim last.
        x = torch.permute(x, (0, 2, 1))
        x, _ = self.pool(x)
        return self.rho(x)
