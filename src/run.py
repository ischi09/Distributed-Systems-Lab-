import click
import numpy as np
import torch

from deepsets.experiments import DeepSetExperiment


@click.command()
@click.option('--random-seed', envvar='SEED', default=42)
def main(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    experiment_type = 'cardinality'
    print(f'Running experiment of type: {experiment_type}')

    experiment = DeepSetExperiment(
        type=experiment_type,
        log_dir='./log',
        lr=1e-3)

    for i in range(5):
        print(f'Training epoch {i}...')
        experiment.train_epoch(i)

    print('\nEvaluating on test set...')
    percentage_correct = experiment.evaluate()

    print(f'\nCorrect: {percentage_correct * 100}%')


if __name__ == '__main__':
    main()
