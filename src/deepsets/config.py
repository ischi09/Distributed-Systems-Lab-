from dataclasses import dataclass


@dataclass
class Paths:
    log: str
    results: str

@dataclass
class Experiment:
    epochs: int
    lr: float
    batch_size: int
    random_seed: int
    loss: str
    weight_decay: float


@dataclass
class Model:
  accumulator: str
  laten_dim: int


@dataclass
class Trainset:
  n_samples: int
  max_set_size: int
  min_value: int
  max_value: int
  label: str
  multisets: bool

@dataclass
class Validset:
  n_samples: int
  max_set_size: int
  min_value: int
  max_value: int
  label: str
  multisets: bool

@dataclass
class Testset:
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
    trainset: Trainset
    validset: Validset
    testset: Testset