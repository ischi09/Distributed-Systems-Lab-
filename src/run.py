import click
import numpy as np
import torch

from deepsets.config import Config, Paths, Trainset
from deepsets.config import Experiment as ExperimentConfig
from deepsets.experiments import Experiment
from deepsets.networks import DeepSetsInvariant, MLP, accumulate_sum
from deepsets.datasets import SetDataset


@click.command()
@click.option("--random-seed", envvar="SEED", default=42)
def main(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    experiment_type = "sum"
    use_multisets = True

    # Set up dataset.
    if experiment_type == "max":

        def label_generator(x: torch.Tensor):
            return x.max()

    elif experiment_type == "mode":

        def label_generator(x: torch.Tensor):
            return torch.squeeze(x).mode().values

    elif experiment_type == "cardinality":

        def label_generator(x: torch.Tensor):
            return torch.tensor(len(x), dtype=torch.float)

    else:

        def label_generator(x: torch.Tensor):
            return x.sum()

    train_set = SetDataset(
        n_samples=10000,
        max_set_size=10,
        min_value=0,
        max_value=10,
        label_generator=label_generator,
        generate_multisets=use_multisets,
    )

    valid_set = SetDataset(
        n_samples=1000,
        max_set_size=10,
        min_value=0,
        max_value=10,
        label_generator=label_generator,
        generate_multisets=use_multisets,
    )

    test_set = SetDataset(
        n_samples=1000,
        max_set_size=10,
        min_value=0,
        max_value=10,
        label_generator=label_generator,
        generate_multisets=use_multisets,
    )

    # Set up model.
    model = DeepSetsInvariant(
        phi=MLP(input_dim=1, hidden_dim=10, output_dim=10),
        rho=MLP(input_dim=10, hidden_dim=10, output_dim=1),
        accumulator=accumulate_sum,
    )

    config = Config(
        Paths("./log", None),
        ExperimentConfig(2, 1e-3, 64, 42, "mse", 5e-4),
        None,
        Trainset(None, None, None, None, experiment_type, use_multisets),
        None,
        None,
    )

    experiment = Experiment(
        config=config,
        model=model,
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
    )

    experiment.run()


if __name__ == "__main__":
    main()
