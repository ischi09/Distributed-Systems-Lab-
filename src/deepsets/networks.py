# Based on https://github.com/yassersouri/pytorch-deep-sets

import torch
import torch.nn as nn


class InvariantDeepSet(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        super().__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Map to latent representation and sum over it.
        sum = torch.zeros((self.phi.output_dim,))
        for elem in x:
            sum += self.phi.forward(elem)

        # Apply final map to produce output.
        return self.rho.forward(sum)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)
