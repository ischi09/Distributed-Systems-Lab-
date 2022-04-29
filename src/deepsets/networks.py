from typing import Callable

import torch
import torch.nn as nn

from .config import Model as ModelConfig
from .config import Dataset as DatasetConfig
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
    neg_infinity = torch.tensor(float("-inf"))
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
            phi=InternalMLP(
                input_dim=model_config.data_dim,
                hidden_dim=10,
                output_dim=model_config.laten_dim,
            ),
            rho=InternalMLP(
                input_dim=model_config.laten_dim,
                hidden_dim=10,
                output_dim=output_dim,
            ),
            accumulator=ACCUMLATORS[model_config.accumulator],
        )
    elif model_config.type == "pna":
        model = PNA(
            mlp=InternalMLP(
                input_dim=12, hidden_dim=10, output_dim=output_dim
            ),
            delta=delta,
        )
    elif model_config.type == "mlp":
        model = MLP(
            input_dim=10,
            hidden_dim=10,
            output_dim=output_dim,
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
            phi=InternalMLP(
                input_dim=model_config.data_dim,
                hidden_dim=10,
                output_dim=model_config.laten_dim,
            ),
            rho=InternalMLP(
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


class InternalMLP(nn.Module):
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


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mlp = InternalMLP(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(dim=-1)
        return self.mlp(x)


class SortedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mlp = InternalMLP(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x, _ = torch.sort(x, dim=1)
        x = x.squeeze(dim=-1)
        return self.mlp(x)


class PNA(nn.Module):
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
