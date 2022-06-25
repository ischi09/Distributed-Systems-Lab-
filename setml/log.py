import os
import sys
import logging

from torch.utils.tensorboard import SummaryWriter

from .config import Config


def get_model_subdir(config: Config) -> str:
    multisets_id = "multisets" if config.task.multisets else "sets"
    return os.path.join(
        config.model.type,
        f"{config.task.label}-{multisets_id}",
        f"lr:{config.experiment.lr}-"
        + f"wd:{config.experiment.weight_decay}-"
        + f"{config.experiment.batch_size}",
    )


def build_summary_writer(config: Config) -> SummaryWriter:
    log_dir = os.path.join(config.paths.log, get_model_subdir(config))
    return SummaryWriter(log_dir=log_dir)
