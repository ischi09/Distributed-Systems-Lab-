from typing import Callable

import torch
import torch.nn as nn

from .config import Model as ModelConfig


def accumulate_sum(x: torch.FloatTensor, mask: torch.FloatTensor) -> torch.FloatTensor:
    x = x * mask
    return x.sum(axis=1)


def accumulate_mean(x: torch.FloatTensor, mask: torch.FloatTensor) -> torch.FloatTensor:
    sum = accumulate_sum(x, mask)
    n_elements = mask.sum(axis=1)
    return sum / n_elements


def accumulate_std(x: torch.FloatTensor, mask: torch.FloatTensor) -> torch.FloatTensor:
    mean = accumulate_mean(x, mask).unsqueeze(dim=-1)
    variance = accumulate_mean(torch.square(x - mean), mask)
    return torch.sqrt(variance)


def accumulate_max(x: torch.FloatTensor, mask: torch.FloatTensor) -> torch.FloatTensor:
    neg_infinity = torch.tensor(float("-inf"))
    x = torch.where(mask.byte(), x, neg_infinity)
    return x.max(dim=1).values


def accumulate_min(x: torch.FloatTensor, mask: torch.FloatTensor) -> torch.FloatTensor:
    return -accumulate_max(-x, mask)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


ACCUMLATORS = {
    "sum": accumulate_sum,
    "mean": accumulate_mean,
    "std": accumulate_std,
    "max": accumulate_max,
    "min": accumulate_min,
}


def generate_model(config: ModelConfig) -> nn.Module:
    model = None
    if config.type == "deepsets_mlp":
        model = DeepSetsInvariant(
            phi=MLP(
                input_dim=config.data_dim, hidden_dim=10, output_dim=config.laten_dim,
            ),
            rho=MLP(
                input_dim=config.laten_dim, hidden_dim=10, output_dim=config.data_dim,
            ),
            accumulator=ACCUMLATORS[config.accumulator],
        )
    return model


class DeepSetsInvariant(nn.Module):
    def __init__(
        self,
        phi: nn.Module,
        rho: nn.Module,
        accumulator: Callable[
            [torch.FloatTensor, torch.FloatTensor], torch.FloatTensor
        ],
    ):
        super().__init__()
        self.phi = phi
        self.rho = rho
        self.accumulator = accumulator

    def forward(
        self, x: torch.FloatTensor, mask: torch.FloatTensor
    ) -> torch.FloatTensor:
        x = self.phi(x)  # x.shape = (batch_size, max_set_size, input_dim)
        x = self.accumulator(x, mask)
        return self.rho(x)


class pna(nn.Module):
    def __init__(self, mlp: nn.Module, unpadded: torch.FloatTensor):
        super().__init__()
        self.mlp = mlp
        self.unpadded = unpadded

    def scale_amplification(self, x: torch.FloatTensor) -> torch.FloatTensor:
        delta = torch.sum(self.unpadded) / torch.size(self.unpadded, dim=1) + 1
        scale = torch.log(self.unpadded + 1) / delta  # TODO is this right??
        return torch.mul(x, scale)

    def scale_attenuation(self, x: torch.FloatTensor) -> torch.FloatTensor:
        delta = torch.sum(self.unpadded) / torch.size(self.unpadded, dim=1) + 1
        scale = delta / torch.log(self.unpadded + 1)  # TODO is this right??
        return torch.mul(x, scale)

    def forward(
        self, x: torch.FloatTensor, mask: torch.FloatTensor
    ) -> torch.FloatTensor:
        # Aggregration
        mean = accumulate_mean(x, mask)
        agg_max = accumulate_max(x, mask)
        agg_min = accumulate_min(x, mask)
        std = accumulate_std(x, mask)

        aggr_concat = torch.concat([mean, agg_max, agg_min, std])

        # Scaling
        identity = aggr_concat
        amplification = self.scale_amplification(aggr_concat)
        attenuation = self.scale_attenuation(aggr_concat)

        scale_concat = torch.concat([identity, amplification, attenuation])

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

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)
