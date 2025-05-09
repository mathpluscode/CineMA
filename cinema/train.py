"""Training script for downstream tasks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from cinema.convvit import load_pretrain_weights, param_groups_lr_decay
from cinema.device import get_amp_dtype_and_device, print_model_info
from cinema.log import get_logger, init_wandb
from cinema.optim import EarlyStopping, GradScaler, adjust_learning_rate, get_n_accum_steps, save_checkpoint

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    import wandb
    from omegaconf import DictConfig
logger = get_logger(__name__)


def maybe_reduce_batch_size(config: DictConfig, n: int) -> DictConfig:
    """Reduce batch size if dataset is too small.

    Args:
        config: configuration file.
        n: number of samples in the dataset.

    Returns:
        config: updated config.
    """
    batch_size = config.train.batch_size
    if n >= batch_size:
        return config
    while n < batch_size:
        batch_size //= 2
    if batch_size == 0:
        raise ValueError(f"Dataset size is too small {n}.")
    logger.warning(f"Using batch size {batch_size} instead.")
    config.train.batch_size = batch_size
    config.train.batch_size_per_device = min(config.train.batch_size_per_device, batch_size)
    return config


def maybe_subset_dataset(
    config: DictConfig,
    train_meta_df: pd.DataFrame,
    val_meta_df: pd.DataFrame,
    group_col: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subset the dataset if needed.

    Args:
        config: configuration file.
        train_meta_df: metadata dataframe for training.
        val_meta_df: metadata dataframe for validation.
        group_col: column for stratified sampling.

    Returns:
        train_meta_df: subset metadata dataframe for training.
        val_meta_df: subset metadata dataframe for validation.
    """
    if config.data.max_n_samples > 0:
        train_ratio = min(config.data.max_n_samples / len(train_meta_df), 1.0)
        val_ratio = min(config.data.max_n_samples / len(val_meta_df), 1.0)
        if group_col:
            train_meta_df = train_meta_df.groupby(group_col).sample(frac=train_ratio, random_state=0)
            val_meta_df = val_meta_df.groupby(group_col).sample(frac=train_ratio, random_state=0)
        else:
            train_meta_df = train_meta_df.sample(frac=train_ratio, random_state=0, ignore_index=True)
            val_meta_df = val_meta_df.sample(frac=val_ratio, random_state=0, ignore_index=True)
        logger.info(f"Using {train_meta_df.shape[0]} samples for training and {val_meta_df.shape[0]} for validation.")
    if config.data.proportion < 1:
        train_meta_df = train_meta_df.sample(
            n=int(config.data.proportion * len(train_meta_df)), random_state=config.seed, ignore_index=True
        )
        logger.info(f"Using {config.data.proportion * 100}% samples, {train_meta_df.shape[0]} samples for training.")
    return train_meta_df, val_meta_df


def train_one_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_scaler: GradScaler,
    amp_dtype: torch.dtype,
    device: torch.device,
    epoch: int,
    n_accum_steps: int,
    n_samples: int,
    config: DictConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None,
    loss_fn: Callable[
        [nn.Module, dict[str, torch.Tensor], list[str], torch.device], tuple[torch.Tensor, dict[str, float]]
    ],
) -> int:
    """Finetune model for one epoch.

    Args:
        model: model to task.
        train_dataloader: dataloader for training data.
        transform: transform for training data.
        optimizer: optimizer for training.
        loss_scaler: GradScaler for mixed precision training.
        amp_dtype: dtype for mixed precision training.
        device: device to use.
        epoch: current epoch.
        n_accum_steps: number of gradient accumulation steps.
        n_samples: number of samples processed.
        config: configuration file.
        wandb_run: wandb run for logging.
        loss_fn: loss function.

    Returns:
        n_samples: number of samples processed.
    """
    model.train()
    views = train_dataloader.dataset.views
    batch_size_per_step = config.train.batch_size_per_device
    clip_grad = config.train.clip_grad if config.train.clip_grad > 0 else None
    for i, batch in enumerate(train_dataloader):
        lr = adjust_learning_rate(
            optimizer=optimizer,
            step=i / len(train_dataloader) + epoch,
            warmup_steps=config.train.n_warmup_epochs,
            max_n_steps=config.train.n_epochs,
            lr=config.train.lr,
            min_lr=config.train.min_lr,
        )
        with torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
            loss, metrics = loss_fn(model, batch, views, device)
            metrics = {f"train_{k}": v for k, v in metrics.items()}

        if torch.isnan(loss).any():
            logger.error(f"Got NaN loss, metrics are {metrics}.")
            continue

        loss /= n_accum_steps
        update_grad = (i + 1) % n_accum_steps == 0
        grad_norm = loss_scaler(
            loss=loss,
            optimizer=optimizer,
            clip_grad=clip_grad,
            parameters=model.parameters(),
            update_grad=update_grad,
        )
        if update_grad:
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        n_samples += batch_size_per_step
        if update_grad and (wandb_run is not None):
            metrics.update(
                {
                    "grad_norm": grad_norm.item(),
                    "lr": lr,
                    "n_samples": n_samples,
                    "epoch": epoch,
                },
            )
            wandb_run.log(metrics)
    return n_samples


def run_train(  # noqa: C901
    config: DictConfig,
    load_dataset: Callable[[DictConfig], tuple[Dataset, Dataset]],
    get_model_fn: Callable[[DictConfig], nn.Module],
    loss_fn: Callable[
        [nn.Module, dict[str, torch.Tensor], list[str], torch.device], tuple[torch.Tensor, dict[str, float]]
    ],
    eval_dataloader_fn: Callable[
        [
            nn.Module,
            DataLoader,
            dict[str, tuple[int, ...]],
            dict[str, tuple[float, ...]],
            torch.dtype,
            torch.device,
        ],
        dict[str, float],
    ],
) -> None:
    """Main function to fine-tune.

    Args:
        config: configuration file.
        load_dataset: function to load the train and val datasets.
        get_model_fn: function to initialize the model.
        loss_fn: loss function for segmentation tasks.
        eval_dataloader_fn: evaluation function for segmentation tasks.
    """
    amp_dtype, device = get_amp_dtype_and_device()
    torch.manual_seed(config.seed)

    # load dataset
    train_dataset, val_dataset = load_dataset(config=config)
    config = maybe_reduce_batch_size(config, len(train_dataset))
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=config.train.batch_size_per_device,
        drop_last=True,
        pin_memory=True,
        num_workers=config.train.n_workers,
    )

    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=1,  # do not batch as each sample has different number of slices
        drop_last=False,
        pin_memory=True,
        num_workers=config.train.n_workers,
    )

    # determine gradient accumulation
    n_accum_steps = get_n_accum_steps(
        batch_size=config.train.batch_size,
        batch_size_per_device=config.train.batch_size_per_device,
        world_size=1,
    )

    # load model
    views = [config.model.views] if isinstance(config.model.views, str) else config.model.views
    model = get_model_fn(config)
    print_model_info(model)
    if config.model.ckpt_path is not None:
        model = load_pretrain_weights(
            model=model, views=views, ckpt_path=Path(config.model.ckpt_path), freeze=config.model.freeze_pretrained
        )
    model.to(device)

    # init wandb
    tags = [
        config.data.name,
        config.model.name,
        *views,
        config.task,
        f"seed{config.seed}",
        f"{int(config.data.proportion * 100)}%",
    ]
    if config.model.ckpt_path is not None:
        tags.append("finetuned")
    if hasattr(config.data, "class_column"):
        tags.append(config.data.class_column)
    if hasattr(config.data, "regression_column"):
        tags.append(config.data.regression_column)
    wandb_run, ckpt_dir = init_wandb(config=config, tags=sorted(set(tags)))

    # init optimizer
    logger.info("Initializing optimizer.")
    if config.model.ckpt_path is not None:
        param_groups = param_groups_lr_decay(
            model,
            no_weight_decay_list=[],
            weight_decay=config.train.weight_decay,
            layer_decay=config.train.layer_decay,
        )
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.AdamW(param_groups, lr=config.train.lr, betas=config.train.betas)
    loss_scaler = GradScaler()

    # train
    logger.info("Start training.")
    patch_size_dict = {v: config.data.sax.patch_size if v == "sax" else config.data.lax.patch_size for v in views}
    spacing_dict = {v: config.data.sax.spacing if v == "sax" else config.data.lax.spacing for v in views}
    early_stop = EarlyStopping(
        min_delta=config.train.early_stopping.min_delta,
        patience=config.train.early_stopping.patience,
    )
    n_samples = 0
    saved_ckpt_paths = []
    for epoch in range(config.train.n_epochs):
        optimizer.zero_grad()
        n_samples = train_one_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            amp_dtype=amp_dtype,
            device=device,
            epoch=epoch,
            n_accum_steps=n_accum_steps,
            n_samples=n_samples,
            config=config,
            wandb_run=wandb_run,
            loss_fn=loss_fn,
        )

        if (ckpt_dir is None) or ((epoch + 1) % config.train.eval_interval != 0):
            continue

        # evaluate model
        logger.info(f"Start evaluating model at epoch {epoch}.")
        model.eval()

        val_metrics = eval_dataloader_fn(
            model,
            val_dataloader,
            patch_size_dict,
            spacing_dict,
            amp_dtype,
            device,
        )
        val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
        val_metrics["n_samples"] = n_samples
        if wandb_run is not None:
            wandb_run.log(val_metrics)
        val_metrics_str = {k: v if isinstance(v, int) else f"{v:.2e}" for k, v in val_metrics.items()}
        logger.info(f"Validation metrics: {val_metrics_str}.")

        # early stopping update
        early_stop_metric = val_metrics[config.train.early_stopping.metric]
        if config.train.early_stopping.mode == "max":
            early_stop_metric = -early_stop_metric
        early_stop.update(early_stop_metric)
        logger.info(
            f"Early stop updated {epoch}: "
            f"should_stop = {early_stop.should_stop}, "
            f"patience_count = {early_stop.patience_count}, "
            f"patience = {early_stop.patience}."
        )

        # save model checkpoint
        if early_stop.has_improved or epoch == 0:
            ckpt_path = save_checkpoint(ckpt_dir, epoch, model, optimizer, loss_scaler, n_samples)
            saved_ckpt_paths.append(ckpt_path)
            logger.info(f"Saved checkpoint of epoch {epoch} at {ckpt_path} after {n_samples} samples.")
            if len(saved_ckpt_paths) > config.train.max_n_ckpts > 0:
                to_delete = saved_ckpt_paths.pop(0)
                to_delete.unlink(missing_ok=True)
                logger.info(f"Deleted an outdated checkpoint {to_delete}.")

        # early stopping
        if early_stop.should_stop:
            logger.info(
                f"Met early stopping criteria with {config.train.early_stopping.metric} = "
                f"{early_stop.best_metric} and patience {early_stop.patience_count}, breaking."
            )
            break
    logger.info(f"Last checkpoint is {saved_ckpt_paths[-1]}")
