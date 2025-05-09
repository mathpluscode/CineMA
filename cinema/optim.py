"""Optimization utilities."""

from __future__ import annotations

import math
from math import inf
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn, optim

from cinema.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
logger = get_logger(__name__)


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    step: float,
    warmup_steps: int,
    max_n_steps: int,
    lr: float,
    min_lr: float,
) -> float:
    """Decay the learning rate with half-cycle cosine after warmup.

    https://github.com/facebookresearch/mae/blob/main/util/lr_sched.py

    Args:
        optimizer: optimizer.
        step: current step.
        warmup_steps: number of warmup steps.
        max_n_steps: total number of steps.
        lr: initial learning rate.
        min_lr: minimum learning rate.
    """
    if step < warmup_steps:
        lr = lr * step / warmup_steps
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (step - warmup_steps) / (max_n_steps - warmup_steps))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def apply_optim_scheduler(
    optimizer: torch.optim.Optimizer, lr: float, last_layer_lr: float, weight_decay: float
) -> None:
    """Apply optimizer scheduler.

    Args:
        optimizer: optimizer.
        lr: learning rate.
        last_layer_lr: learning rate for the last layer.
        weight_decay: weight decay.
    """
    for param_group in optimizer.param_groups:
        param_group["weight_decay"] = weight_decay * param_group["weight_decay_scale"]
        param_group["lr"] = (last_layer_lr if param_group["is_last_layer"] else lr) * param_group["lr_scale"]


class CosineScheduler:
    """Cosine scheduler with warmup and freeze.

    https://github.com/facebookresearch/dinov2/blob/main/dinov2/utils/utils.py
    """

    def __init__(
        self,
        base_value: float,
        final_value: float,
        total_iters: int,
        warmup_iters: int = 0,
        start_warmup_value: float = 0.0,
        freeze_iters: int = 0,
    ) -> None:
        """Initialize the scheduler.

        Args:
            base_value: base value.
            final_value: final value or after total_iters.
            total_iters: total number of iterations.
            warmup_iters: number of warmup iterations.
            start_warmup_value: initial value for warmup.
            freeze_iters: number of freeze iterations.
        """
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters,))
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        if len(self.schedule) != self.total_iters:
            raise ValueError(
                f"Length of schedule {len(self.schedule)} should be equal to total_iters {self.total_iters}."
            )

    def __getitem__(self, it: int) -> float | np.ndarray:
        """Get the value at iteration.

        Args:
            it: iteration.
        """
        if it >= self.total_iters:
            return self.final_value
        return self.schedule[it]


def get_n_accum_steps(
    batch_size: int,
    batch_size_per_device: int,
    world_size: int,
) -> int:
    """Get the number of gradient accumulation steps.

    Args:
        batch_size: effective batch size.
        batch_size_per_device: batch size per device.
        world_size: number of devices.
    """
    batch_size_per_step = batch_size_per_device * world_size
    logger.info(f"batch_size_per_step = {batch_size_per_device} x {world_size} = {batch_size_per_step}")
    if batch_size_per_step > batch_size:
        raise ValueError(f"batch_size_per_step {batch_size_per_step} should be less than batch_size {batch_size}.")
    if batch_size % batch_size_per_step != 0:
        raise ValueError(f"batch_size {batch_size} should be divisible by batch_size_per_step {batch_size_per_step}.")
    n_accum_steps = batch_size // batch_size_per_step
    if n_accum_steps > 1:
        logger.info(f"gradient accumulation every {n_accum_steps} iterations so that total batch_size is {batch_size}.")
    return n_accum_steps


def get_grad_norm(parameters: torch.Tensor | Iterable[torch.Tensor], norm_type: float = 2.0) -> torch.Tensor:
    """Calculate the norm of gradients.

    Args:
        parameters: model parameters.
        norm_type: type of the used norm.

    Returns:
        norm: norm of the gradients.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type,
        )
    return total_norm


class GradScaler:
    """Gradient scaler with gradient norm clip."""

    state_dict_key = "amp_scaler"

    def __init__(self) -> None:
        """Initialize the scaler."""
        self._scaler = torch.GradScaler("cuda", enabled=torch.cuda.is_available())

    def __call__(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        clip_grad: float | None = None,
        parameters: torch.Tensor | Iterable[torch.Tensor] | None = None,
        create_graph: bool = False,
        update_grad: bool = True,
    ) -> torch.Tensor:
        """Backward pass with gradient scaling.

        Args:
            loss: loss value.
            optimizer: optimizer.
            clip_grad: gradient clipping value.
            parameters: model parameters.
            create_graph: whether to create graph.
            update_grad: whether to update gradients.

        Returns:
            norm: gradient norm.
        """
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if parameters is None:
                raise ValueError("parameters must not be None.")
            if clip_grad is not None:
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self) -> dict[str, Any]:
        """Return the state dict of the scaler."""
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dict of the scaler."""
        self._scaler.load_state_dict(state_dict)


def save_checkpoint(
    ckpt_dir: Path,
    epoch: int,
    model_wo_ddp: nn.Module,
    optimizer: optim.Optimizer,
    loss_scaler: GradScaler,
    n_samples: int,
) -> Path:
    """Save checkpoint.

    Args:
        ckpt_dir: checkpoint directory.
        epoch: current epoch.
        model_wo_ddp: model without DDP.
        optimizer: optimizer.
        loss_scaler: loss scaler.
        n_samples: number of samples processed.

    Returns:
        ckpt_path: checkpoint path.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"ckpt_{epoch}.pt"
    to_save = {
        "model": model_wo_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scaler": loss_scaler.state_dict(),
        "n_samples": n_samples,
    }
    torch.save(to_save, ckpt_path)
    return ckpt_path


def load_checkpoint_and_optimizer(
    ckpt_path: Path,
    model_wo_ddp: nn.Module,
    optimizer: optim.Optimizer,
    loss_scaler: GradScaler,
) -> tuple[nn.Module, optim.Optimizer, GradScaler, int, int]:
    """Load checkpoint and optimizer.

    Args:
        ckpt_path: checkpoint path.
        model_wo_ddp: model without DDP.
        optimizer: optimizer.
        loss_scaler: loss scaler.

    Returns:
        model_wo_ddp: model without DDP.
        optimizer: optimizer.
        loss_scaler: loss scaler.
        epoch: epoch to resume from.
        n_samples: number of samples.
    """
    logger.info(f"Loading checkpoint from {ckpt_path}.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_wo_ddp.load_state_dict(ckpt["model"])
    logger.info(f"Loaded checkpoint from {ckpt_path}.")

    optimizer.load_state_dict(ckpt["optimizer"])
    loss_scaler.load_state_dict(ckpt["scaler"])
    logger.info(f"Loaded optimizer from {ckpt_path}.")

    return model_wo_ddp, optimizer, loss_scaler, ckpt["epoch"], ckpt.get("n_samples", 0)


class EarlyStopping:
    """Early stopping to avoid overfitting during training, by monitoring a metric that should be minimized."""

    def __init__(
        self,
        min_delta: float,
        patience: int,
    ) -> None:
        """Initialize the early stopping.

        Args:
            min_delta: minimum change in the monitored quantity to qualify as an improvement.
            patience: number of epochs with no improvement after which training will be stopped.
        """
        self.min_delta = min_delta
        self.best_metric = float("inf")
        self.patience = patience
        self.patience_count = 0
        self.should_stop = False
        self.has_improved = False

    def update(self, metric: float) -> None:
        """Update the state based on metric.

        Args:
            metric: metric to compare.
        """
        self.has_improved = self.best_metric > metric  # not necessarily improved enough
        if self.has_improved and self.best_metric >= metric + self.min_delta:
            self.best_metric = metric
            self.patience_count = 0
        else:
            self.patience_count += 1
            self.should_stop = self.patience_count >= self.patience
