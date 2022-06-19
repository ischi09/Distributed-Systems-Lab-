from dataclasses import dataclass


@dataclass
class Paths:
    log: str
    checkpoints: str
    results: str


@dataclass
class Experiment:
    max_epochs: int
    patience: int
    min_delta: float
    lr: float
    batch_size: int
    random_seed: int
    weight_decay: float
    grad_norm_threshold: float
    lr_scheduler_factor: float
    lr_scheduler_patience: int
    use_batch_sampler: bool
    use_gpu: bool
    results_out: str


@dataclass
class Model:
    type: str
    data_dim: int
    laten_dim: int


@dataclass
class Task:
    max_set_size: int
    min_value: int
    max_value: int
    label: str
    multisets: bool


@dataclass
class Datasets:
    train_samples: int
    valid_samples: int
    test_samples: int


@dataclass
class Config:
    paths: Paths
    experiment: Experiment
    model: Model
    task: Task
    datasets: Datasets
