"""Distributed training.

https://github.com/facebookresearch/mae/blob/main/util/misc.py
https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py
"""

import datetime
import os
import socket
from contextlib import closing

import torch
from torch import nn
from torch.backends import cudnn
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from cinema.log import get_logger

logger = get_logger(__name__)


def get_free_port() -> int:
    """Return a free port.

    Returns:
        port: a free port.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def ddp_setup(rank: int, world_size: int, port: int) -> None:
    """Distributed data parallel setup.

    Args:
        rank: Unique identifier of each process.
        world_size: Total number of processes.
        port: Port number for the master process.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{port}"
    logger.info("Setting MASTER_ADDR=localhost.")
    logger.info(f"Setting MASTER_PORT={port}.")
    init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=5400))
    torch.cuda.set_device(rank)


def get_amp_dtype_and_device() -> tuple[torch.dtype, torch.device]:
    """Get automatic mixed precision dtype and device.

    Returns:
        amp_dtype: automatic mixed precision dtype.
        device: device.
    """
    amp_dtype = torch.float16
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        # enable cuDNN auto-tuner, https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
        cudnn.benchmark = True
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            logger.info("Using bfloat16 for automatic mixed precision.")
    else:
        logger.info("CUDA is not available, using CPU.")
        device = torch.device("cpu")

    return amp_dtype, device


def print_model_info(model: nn.Module) -> None:
    """Print model information.

    Args:
        model: Model to print information.
    """
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"number of parameters: {n_params:,}")
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of trainable parameters: {n_trainable_params:,}")


def setup_ddp_model(model: nn.Module, device: torch.device, rank: int, world_size: int) -> tuple[nn.Module, nn.Module]:
    """Setup the model for distributed training.

    Args:
        model: Model to setup.
        device: Device to use.
        rank: Unique identifier of each process.
        world_size: Total number of processes.

    Returns:
        model: Potential distributed model.
        model_wo_ddp: Model without DistributedDataParallel.
    """
    model.to(device)
    model_wo_ddp = model
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
        model_wo_ddp = model.module
    return model, model_wo_ddp
