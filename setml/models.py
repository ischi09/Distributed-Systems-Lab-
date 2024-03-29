from typing import Callable, Union

import sklearn
from sklearn.dummy import DummyRegressor
import torch
import torch.nn as nn

from .config import Config
from .tasks import ClassificationTask, get_task
from .set_transformer.modules import SAB, PMA, ISAB
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
        drop_prob=0.2,  # TODO: Unused.
    ):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mask) -> torch.Tensor:
        h0 = torch.zeros(
            self.n_layers, x.size(0), self.hidden_dim
        ).requires_grad_()

        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            x,
            mask.sum(axis=1).flatten().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed_input, h0.detach())

        out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )

        final = self.fc(out[:, -1, :])

        return final


class SortedGRU(GRUNet):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int
    ):
        super(SortedGRU, self).__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
        )

    def forward(self, x, mask) -> torch.Tensor:
        x, _ = torch.sort(x, dim=1)
        return super(SortedGRU, self).forward(x, mask)


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

    def forward(self, x, mask) -> torch.Tensor:
        h0 = torch.zeros(
            self.n_layers, x.size(0), self.hidden_dim
        ).requires_grad_()

        c0 = torch.zeros(
            self.n_layers, x.size(0), self.hidden_dim
        ).requires_grad_()

        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            x,
            mask.sum(axis=1).flatten().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, (_, _) = self.lstm(
            packed_input, (h0.detach(), c0.detach())
        )

        out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )

        return self.fc(out[:, -1, :])


class SortedLSTM(LSTMNet):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        drop_prob=0.2,
    ):
        super(SortedLSTM, self).__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            drop_prob=drop_prob,
        )

    def forward(self, x, mask) -> torch.Tensor:
        x, _ = torch.sort(x, dim=1)
        return super(SortedLSTM, self).forward(x, mask)


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


class PnaDs1t(nn.Module):
    def __init__(self, output_dim: int, delta: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(12, 30),
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


class SetTransformerDs1t(nn.Module):
    """
    For optimal results, use with the following settings:

    batch_size: 16
    use_batch_sampler: True
    latent_dim: 64
    """

    def __init__(
        self,
        input_dim: int,
        num_outputs: int,
        output_dim: int,
        num_inds: int = 32,
        hidden_dim: int = 128,
        num_heads: int = 4,
        use_layer_norm: bool = False,
    ):
        super(SetTransformerDs1t, self).__init__()

        self.enc = nn.Sequential(
            SAB(input_dim, hidden_dim, num_heads, ln=use_layer_norm),
            SAB(hidden_dim, hidden_dim, num_heads, ln=use_layer_norm),
        )

        self.dec = nn.Sequential(
            PMA(hidden_dim, num_heads, num_outputs, ln=use_layer_norm),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.enc(x)
        x = self.dec(x)
        x = x.squeeze(dim=-1)
        return x


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
    elif model_config.type == "lstm":
        model = LSTMNet(
            input_dim=config.model.data_dim,
            hidden_dim=config.model.latent_dim,
            output_dim=output_dim,
            n_layers=9,
        )
    elif model_config.type == "sorted_lstm":
        model = SortedLSTM(
            input_dim=config.model.data_dim,
            hidden_dim=config.model.latent_dim,
            output_dim=output_dim,
            n_layers=9,
        )
    elif model_config.type == "gru":
        model = GRUNet(
            input_dim=config.model.data_dim,
            hidden_dim=config.model.latent_dim,
            output_dim=output_dim,
            n_layers=9,
        )
    elif model_config.type == "sorted_gru":
        model = SortedGRU(
            input_dim=config.model.data_dim,
            hidden_dim=config.model.latent_dim,
            output_dim=output_dim,
            n_layers=2,
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
    elif model_config.type == "pna_ds1t":
        model = PnaDs1t(
            output_dim=output_dim,
            delta=delta,
        )
    elif model_config.type == "set_transformer_ds1t":
        model = SetTransformerDs1t(
            input_dim=config.model.data_dim,
            output_dim=config.model.data_dim,
            hidden_dim=config.model.latent_dim,
            num_heads=8,
            num_outputs=1,
        )
    elif model_config.type == "mean_baseline":
        model = DummyRegressor()
    return model
