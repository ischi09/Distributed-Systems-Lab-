import click
from isort import Config
import numpy as np
import hydra
import torch

from deepsets.experiments import DeepSetExperiment


@click.command()
@click.option('--random-seed', envvar='SEED', default=42)


@hydra.main(config_path="conf", config_name="config")
def main(cfg:Config):
    np.random.seed(cfg.experiment.random_seed)
    torch.manual_seed(cfg.experiment.random_seed)
    torch.cuda.manual_seed_all(cfg.experiment.random_seed)

    experiment_type = cfg.label # TODO Figure out how to access the default value
    use_multisets = cfg.multisets
    print(
        f"Running experiment of type '{experiment_type}' with {'multisets' if use_multisets else 'sets'}")

    experiment = DeepSetExperiment(
        type=experiment_type,
        use_multisets=use_multisets,
        log_dir=cfg.paths.log,
        lr=cfg.experiment.lr)

    for i in range(10):
        print(f'Training epoch {i}...')
        experiment.train_epoch(i)

    print('\nEvaluating on test set...')
    percentage_correct = experiment.evaluate()

    print(
        f'\nCorrect (absolute difference < 0.1): {percentage_correct * 100}%')


if __name__ == '__main__':
    main()
