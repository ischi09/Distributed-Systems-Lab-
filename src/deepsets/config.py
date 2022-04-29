from dataclasses import dataclass


@dataclass
class Paths:
    log: str
    models: str
    results: str


@dataclass
class Experiment:
    max_epochs: int
    patience: int
    min_delta: float
    lr: float
    batch_size: int
    random_seed: int
    loss: str
    weight_decay: float
    grad_norm_threshold: float
    use_batch_sampler: bool


@dataclass
class Model:
    type: str
    data_dim: int
    laten_dim: int
    accumulator: str


@dataclass
class Dataset:
    n_samples: int
    max_set_size: int
    min_value: int
    max_value: int
    label: str
    multisets: bool


@dataclass
class Config:
    paths: Paths
    experiment: Experiment
    model: Model
    trainset: Dataset
    validset: Dataset
    testset: Dataset
