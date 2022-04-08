from typing import Callable

import math
import torch
import torch.nn as nn

from .config import Model as ModelConfig
from .set_transformer.modules import SAB, PMA


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


ACCUMLATORS = {
    "sum": lambda x: x.sum(dim=1),
    "mean": lambda x: x.mean(dim=1),
    "std": lambda x: x.std(dim=1),
    "max": lambda x: x.max(dim=1).values,
    "min": lambda x: x.min(dim=1).values,
}


def generate_model(config: ModelConfig, delta: float) -> nn.Module:
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
    elif config.type == "pna":
        model = PNA(
            mlp=MLP(input_dim=12, hidden_dim=10, output_dim=config.data_dim),
            delta=delta,
        )
    elif config.type == "sorted_mlp":
        model = SortedMLP(
            input_dim=10,
            hidden_dim=10,
            output_dim=config.data_dim,
        )
    elif config.type == "small_set_transformer":
        model = SmallSetTransformer()
    return model


class DeepSetsInvariant(nn.Module):
    def __init__(
        self,
        phi: nn.Module,
        rho: nn.Module,
        accumulator: Callable[[torch.FloatTensor], torch.FloatTensor],
    ):
        super().__init__()
        self.phi = phi
        self.rho = rho
        self.accumulator = accumulator

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.phi(x)  # x.shape = (batch_size, set_size, input_dim)
        x = self.accumulator(x)
        return self.rho(x)


class PNA(nn.Module):
    def __init__(self, mlp: nn.Module, delta: float):
        super().__init__()
        self.mlp = mlp
        self.delta = delta

    def scale_amplification(self, x: torch.FloatTensor) -> torch.FloatTensor:
        scale = math.log(x.size(dim=1) + 1) / self.delta
        return x * scale

    def scale_attenuation(self, x: torch.FloatTensor) -> torch.FloatTensor:
        scale = self.delta / math.log(x.size(dim=1) + 1)
        return torch.mul(x, scale)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Aggregration
        mean = ACCUMLATORS["mean"](x)
        agg_max = ACCUMLATORS["max"](x)
        agg_min = ACCUMLATORS["min"](x)
        std = ACCUMLATORS["std"](x)

        aggr_concat = torch.cat((mean, agg_max, agg_min, std), dim=1)

        # Scaling
        identity = aggr_concat
        amplification = self.scale_amplification(aggr_concat)
        attenuation = self.scale_attenuation(aggr_concat)

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

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
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

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x, _ = torch.sort(x, dim=1)
        x = x.squeeze(dim=-1)
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

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(-1)
