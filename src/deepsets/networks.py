from typing import Callable

import torch
import torch.nn as nn

from .config import Model as ModelConfig
from .set_transformer.modules import SAB, PMA


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
                input_dim=config.data_dim,
                hidden_dim=10,
                output_dim=config.laten_dim,
            ),
            rho=MLP(
                input_dim=config.laten_dim,
                hidden_dim=10,
                output_dim=config.data_dim,
            ),
            accumulator=ACCUMLATORS[config.accumulator],
        )
    elif config.type == "small_set_transformer":
        model = SmallSetTransformer()
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


class SmallSetTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=1, dim_out=64, num_heads=4),
            SAB(dim_in=64, dim_out=64, num_heads=4),
        )
        self.dec = nn.Sequential(
            PMA(dim=64, num_heads=4, num_seeds=1),
            nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, x, mask):
        # print(mask.shape)
        # print(mask.sum(dim=1))
        # assert False
        # set_length = int(mask.sum().int())
        # x = x[:, :set_length, :]
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(-1)
