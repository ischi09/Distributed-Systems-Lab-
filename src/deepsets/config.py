from dataclasses import dataclass


@dataclass
class Paths:
    log: str
    models: str
    results: str


@dataclass
class Experiment:
    max_epochs: int
    early_stopping_patience: int
    lr: float
    batch_size: int
    random_seed: int
    loss: str
    weight_decay: float


@dataclass
class Model:
    type: str
    data_dim: int
    laten_dim: int
    accumulator: str


@dataclass
class Dataset:
    max_set_size: int
    min_value: int
    max_value: int
    label: str
    multisets: bool

@dataclass
class Sizeset:
    n_samples: int


@dataclass
class Config:
    paths: Paths
    experiment: Experiment
    model: Model
    set_vals: Dataset
    trainset: Sizeset
    validset: Sizeset
    testset: Sizeset
