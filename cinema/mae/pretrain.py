"""Script for UKB pretraining.

https://github.com/facebookresearch/mae/blob/main/main_pretrain.py
https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from monai.data import Dataset
from monai.transforms import (
    Compose,
    RandZoomd,
    ScaleIntensityd,
    SpatialPadd,
    Transform,
)
from timm.optim import param_groups_weight_decay
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Subset

from cinema import UKB_N_FRAMES
from cinema.device import ddp_setup, get_amp_dtype_and_device, get_free_port, print_model_info, setup_ddp_model
from cinema.log import get_logger, init_wandb
from cinema.mae.mae import CineMA, get_model
from cinema.optim import (
    GradScaler,
    adjust_learning_rate,
    get_n_accum_steps,
    load_checkpoint_and_optimizer,
    save_checkpoint,
)

if TYPE_CHECKING:
    import wandb
    from omegaconf import DictConfig
logger = get_logger(__name__)


def scan_manifests(data_dirs: str | Path | list[str | Path], rescan: bool) -> list[Path]:
    """Scan the manifest files for the dataset.

    Args:
        data_dirs: directory of the dataset, file paths are data_dir/group/eid/*.nii.gz.
        rescan: whether to rescan the data directories.

    Returns:
        manifest_paths: list of manifest file paths.
    """
    if isinstance(data_dirs, Path | str):
        data_dirs = [data_dirs]
    data_dirs = [Path(x) for x in data_dirs]
    manifest_paths = []
    if not rescan:
        for data_dir in data_dirs:
            json_path = Path(data_dir) / "manifest_paths.json"
            if not json_path.exists():
                rescan = True
                break
            with Path.open(json_path, encoding="utf-8") as f:
                manifest_paths_toadd = [Path(x) for x in json.load(f)]
            if not manifest_paths_toadd[0].exists():
                rescan = True
                logger.warning(f"Manifest file paths in {json_path} do not exist, perform scanning manually.")
                break
            manifest_paths += manifest_paths_toadd
    if rescan:
        for data_dir in data_dirs:
            logger.info(f"Start to scan {data_dir} for manifest files.")
            manifest_paths += list(Path(data_dir).glob("**/*_manifest_sax.csv"))
            json_path = Path(data_dir) / "manifest_paths.json"
            with Path.open(json_path, "w", encoding="utf-8") as f:
                json.dump([str(x) for x in manifest_paths], f)
            logger.info(f"Finished scanning {data_dir} for manifest files, file paths saved into {json_path}.")
    logger.info(f"Found in total {len(manifest_paths)} manifest files.")
    return sorted(manifest_paths)


def ukb_load_sample(manifest_path: Path, t: int) -> dict[str, np.ndarray]:
    """Load one UK Biobank data sample.

    Images have been processed to have the same spatial size (x, y) and the same number of time frames (t).

    Args:
        manifest_path: path to the manifest file.
        t: time frame.

    Returns:
        data_dict: dictionary of numpy arrays.
            2C/3C/4C: (x, y, t)
            SAX: (x, y, z, t)
    """
    eid_dir = manifest_path.parent
    eid = eid_dir.name
    data = {}
    reader = sitk.ImageFileReader()
    for view in ["lax_2c", "lax_3c", "lax_4c", "sax"]:
        reader.SetFileName(str(eid_dir / f"{eid}_{view}.nii.gz"))
        reader.ReadImageInformation()
        size = list(reader.GetSize())
        if t >= size[-1]:  # just in case, use the middle frame
            t = size[-1] // 2
        size[-1] = 1
        if view == "sax":
            reader.SetExtractIndex([0, 0, 0, t])
        else:
            reader.SetExtractIndex([0, 0, t])
        reader.SetExtractSize(size)
        data[view] = np.transpose(sitk.GetArrayFromImage(reader.Execute()))[..., 0]
    return data


class UKBDataset(Dataset):
    """UKB dataset, knowing each sample has 50 frames."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Initialise UKB dataset.

        Args:
            args: arguments.
            kwargs: keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng()

    def _transform(self, index: int) -> dict[str, torch.Tensor]:
        """Fetch single data item from `self.data`.

        self.data is list of manifest_paths.

        Args:
            index: index of the data item.

        Returns:
            Transformed images for one time frame.
        """
        t = int(self.rng.integers(UKB_N_FRAMES))
        np_data_i = ukb_load_sample(self.data[index], t)
        data_i = {
            "sax": torch.from_numpy(np_data_i["sax"][None, ...]),
            "lax_2c": torch.from_numpy(np_data_i["lax_2c"][None, ...]),
            "lax_3c": torch.from_numpy(np_data_i["lax_3c"][None, ...]),
            "lax_4c": torch.from_numpy(np_data_i["lax_4c"][None, ...]),
        }
        return self.transform(data_i)


def get_transform(config: DictConfig) -> Transform:
    """Get transform from config.

    Args:
        config: config.

    Returns:
        Transform
    """
    return Compose(
        [
            RandZoomd(
                keys="sax",
                prob=config.transform.prob,
                mode="trilinear",
                padding_mode="constant",
                lazy=True,
                allow_missing_keys=True,
            ),
            RandZoomd(
                keys=("lax_2c", "lax_3c", "lax_4c"),
                prob=config.transform.prob,
                mode="bicubic",
                padding_mode="constant",
                lazy=True,
                allow_missing_keys=True,
            ),
            ScaleIntensityd(keys=("sax", "lax_2c", "lax_3c", "lax_4c")),
            SpatialPadd(
                keys="sax",
                spatial_size=config.data.sax.patch_size,
                method="end",
                lazy=True,
                allow_missing_keys=True,
            ),
            SpatialPadd(
                keys=("lax_2c", "lax_3c", "lax_4c"),
                spatial_size=config.data.lax.patch_size,
                method="end",
                lazy=True,
                allow_missing_keys=True,
            ),
        ]
    )


def pretrain_one_epoch(
    model: CineMA,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_scaler: GradScaler,
    n_accum_steps: int,
    world_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    config: DictConfig,
    epoch: int,
    n_samples: int,
    wandb_run: wandb.sdk.wandb_run.Run | None,
) -> int:
    """Train one epoch.

    Args:
        model: model.
        dataloader: dataloader.
        optimizer: optimizer.
        loss_scaler: loss scaler.
        n_accum_steps: number of accumulation steps.
        world_size: total number of devices.
        device: device.
        amp_dtype: dtype for automatic mixed precision.
        config: config.
        epoch: current epoch.
        n_samples: number of samples trained on since beginning.
        wandb_run: wandb run.

    Returns:
        n_samples: number of samples trained on since beginning.
    """
    views = config.model.views
    batch_size_per_device = config.train.batch_size_per_device
    batch_size_per_step = batch_size_per_device * world_size
    enc_mask_ratio = config.train.enc_mask_ratio
    clip_grad = config.train.clip_grad if config.train.clip_grad > 0 else None

    for i, batch in enumerate(dataloader):
        lr = adjust_learning_rate(
            optimizer=optimizer,
            step=i / len(dataloader) + epoch,
            warmup_steps=config.train.n_warmup_epochs,
            max_n_steps=config.train.n_epochs,
            lr=config.train.lr,
            min_lr=config.train.min_lr,
        )
        with torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
            loss, _, _, metrics = model({k: v.to(device) for k, v in batch.items()}, enc_mask_ratio)
        metrics = {k: v.item() for k, v in metrics.items()}

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
            prefix = f"{views[0]}_" if len(views) == 1 else ""
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
            metrics.update(
                {
                    "grad_norm": grad_norm.item(),
                    "lr": lr,
                    "n_samples": n_samples,
                },
            )
            wandb_run.log(metrics)

    return n_samples


def pretrain(  # noqa: C901, pylint:disable=too-many-statements,too-many-branches
    rank: int,
    world_size: int,
    port: int,
    config: DictConfig,
) -> None:
    """Train the model.

    Args:
        rank: Unique identifier of each process.
        world_size: Total number of processes.
        port: an available port for distributed training.
        config: config for pre-training.
    """
    if world_size > 1:
        ddp_setup(rank, world_size, port)
    amp_dtype, device = get_amp_dtype_and_device()

    # fix the seed for reproducibility
    seed = config.seed + rank
    torch.manual_seed(seed)

    # determine gradient accumulation
    n_accum_steps = get_n_accum_steps(
        batch_size=config.train.batch_size,
        batch_size_per_device=config.train.batch_size_per_device,
        world_size=world_size,
    )

    # load dataset
    manifest_paths = scan_manifests(config.data.dir, rescan=False)
    transform = get_transform(config)
    dataset = UKBDataset(data=manifest_paths, transform=transform)
    if config.data.max_n_samples > 0:
        n = min(config.data.max_n_samples, len(manifest_paths))
        dataset = Subset(dataset, np.arange(n))
        logger.info(f"Using {n} samples for training.")
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = RandomSampler(dataset)
    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=config.train.batch_size_per_device,
        drop_last=True,
        pin_memory=True,
        num_workers=config.train.n_workers_per_device,
    )

    # init models
    model = get_model(config)
    print_model_info(model)
    model, model_wo_ddp = setup_ddp_model(model=model, device=device, rank=rank, world_size=world_size)

    if world_size > 1:
        # sync model weights
        tmp_ckp_path = tempfile.gettempdir() + "/ckpt.pt"
        if rank == 0:
            # All processes should see same parameters as they all start from same
            # random parameters and gradients are synchronized in backward passes.
            # Therefore, saving it in one process is sufficient.
            torch.save(model.state_dict(), tmp_ckp_path)
            logger.info(f"Saved model weights from process 0 to {tmp_ckp_path}.")

        # Use a barrier() to make sure that other process loads the model after process 0 saves it
        dist.barrier()
        # configure map_location properly
        if rank > 0:
            map_location = {"cuda:0": f"cuda:{rank}"}
            model.load_state_dict(torch.load(tmp_ckp_path, map_location=map_location, weights_only=True))
            logger.info(f"Loaded model weights from process 0 to process {rank}.")

    # init optimizer, following timm: set wd as 0 for bias and norm layers
    logger.info("Initializing optimizer.")
    param_groups = param_groups_weight_decay(model=model_wo_ddp, weight_decay=config.train.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.train.lr, betas=config.train.betas)
    loss_scaler = GradScaler()

    # load checkpoint
    n_samples = 0
    start_epoch = 0
    if config.train.ckpt_path is not None:
        model_wo_ddp, optimizer, loss_scaler, epoch, n_samples = load_checkpoint_and_optimizer(
            ckpt_path=config.train.ckpt_path,
            model_wo_ddp=model_wo_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
        )
        logger.info(f"Loaded checkpoint from {config.train.ckpt_path} at epoch {epoch}.")
        start_epoch = epoch + 1

    # init wandb
    wandb_run, ckpt_dir = None, None
    if rank == 0:
        tags = ["ukb_mae_pretrain"]
        if len(config.model.views) > 1:
            tags.append("multi_view")
        wandb_run, ckpt_dir = init_wandb(config=config, tags=tags)

    # train
    logger.info("Start training.")
    saved_ckpt_paths = []
    max_n_ckpts = config.train.max_n_ckpts
    model.train(True)
    for epoch in range(start_epoch, config.train.n_epochs):
        optimizer.zero_grad()
        n_samples = pretrain_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            n_accum_steps=n_accum_steps,
            world_size=world_size,
            device=device,
            amp_dtype=amp_dtype,
            config=config,
            epoch=epoch,
            n_samples=n_samples,
            wandb_run=wandb_run,
        )

        if rank != 0 or ckpt_dir is None:
            # save checkpoint at main process
            continue
        ckpt_path = save_checkpoint(ckpt_dir, epoch, model_wo_ddp, optimizer, loss_scaler, n_samples)
        saved_ckpt_paths.append(ckpt_path)
        logger.info(f"Saved checkpoint of epoch {epoch} at {ckpt_path} after {n_samples} samples.")

        if len(saved_ckpt_paths) <= max_n_ckpts or max_n_ckpts <= 0:
            # delete outdated checkpoints when reaching max_n_ckpts
            continue
        to_delete = saved_ckpt_paths.pop(0)
        ckpt_epoch = int(to_delete.stem.split("_")[1])
        if (ckpt_epoch + 1) % 100 == 0:
            # keep checkpoints at every 100 epochs
            continue
        to_delete.unlink(missing_ok=True)
        logger.info(f"Deleted an outdated checkpoint {to_delete}.")

    if world_size > 1:
        dist.destroy_process_group()


@hydra.main(version_base=None, config_path="", config_name="config")
def main(config: DictConfig) -> None:
    """Entrypoint for pretraining.

    Args:
        config: config loaded from yaml.
    """
    world_size = torch.cuda.device_count()
    port = get_free_port()
    if world_size > 1 and config.ddp:
        mp.spawn(pretrain, args=(world_size, port, config), nprocs=world_size)
    else:
        if world_size > 1:
            logger.warning(f"world_size = {world_size} > 1 but not using DDP.")
        pretrain(rank=0, world_size=1, port=port, config=config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
