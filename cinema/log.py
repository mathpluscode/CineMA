"""Logging utilities."""

import logging
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = "%(threadName)s-%(process)s | %(asctime)s-%(name)s-%(funcName)s:%(lineno)d | [%(levelname)s] %(message)s"
        formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:  # type:ignore[type-arg]
    """Flat a nested dict.

    Args:
        d: dict to flat.
        parent_key: key of the parent.
        sep: separation string.

    Returns:
        flatten dict.
    """
    items = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(d=v, parent_key=new_key, sep=sep))
        else:
            items[new_key] = v
    return dict(items)


def init_wandb(config: DictConfig, tags: list[str]) -> tuple:  # type:ignore[type-arg]
    """Initialize wandb.

    Args:
        config: configuration, assume having config.logging.wandb.project and config.logging.wandb.entity.
        tags: tags for the run, in addition to the model size.

    Returns:
        wandb run and checkpoint directory.
    """
    if config.logging.wandb.project:
        import wandb  # lazy import

        wandb_run = wandb.init(
            project=config.logging.wandb.project,
            entity=config.logging.wandb.entity,
            config=flatten_dict(dict(config)),
            tags=tags,
        )
        if config.logging.dir is None:
            # wandb/run-xxx/files
            ckpt_dir = Path(wandb_run.settings.files_dir).parent / "ckpt"
        else:
            ckpt_dir = Path(config.logging.dir)
            ckpt_dir = ckpt_dir / Path(wandb_run.settings.files_dir).parent.name
            ckpt_dir = Path(ckpt_dir) / "ckpt"
    else:
        wandb_run = None
        ckpt_dir = Path(config.logging.dir) / "ckpt" if config.logging.dir else Path("ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=ckpt_dir / "config.yaml")
    return wandb_run, ckpt_dir
