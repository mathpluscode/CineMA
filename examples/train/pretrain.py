"""A minimalist script to perform pretraining on UKB dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from monai.data import Dataset
from monai.transforms import (
    Compose,
    RandZoomd,
    ScaleIntensityd,
    SpatialPadd,
)
from safetensors.torch import save_file
from timm.optim import param_groups_weight_decay
from torch.utils.data import DataLoader, RandomSampler

from cinema import UKB_N_FRAMES
from cinema.device import get_amp_dtype_and_device
from cinema.log import get_logger
from cinema.mae.mae import CineMA, get_model
from cinema.optim import (
    GradScaler,
    adjust_learning_rate,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig
logger = get_logger(__name__)


class UKBDataset(Dataset):
    """UKB dataset, knowing each sample has 50 frames."""

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data) * UKB_N_FRAMES

    def _transform(self, index: int) -> dict[str, torch.Tensor]:
        """Fetch single data item from `self.data`.

        self.data is list of manifest_paths.

        Args:
            index: index of the data item.

        Returns:
            Transformed images for one time frame.
        """
        path_index = int(index // UKB_N_FRAMES)
        t = int(index % UKB_N_FRAMES)

        eid_dir = self.data[path_index].parent
        data = {}
        reader = sitk.ImageFileReader()
        for view in ["lax_2c", "lax_3c", "lax_4c", "sax"]:
            reader.SetFileName(str(eid_dir / f"{eid_dir.name}_{view}.nii.gz"))
            reader.ReadImageInformation()
            size = list(reader.GetSize())
            size[-1] = 1
            reader.SetExtractIndex([0, 0, 0, t])
            reader.SetExtractSize(size)
            image = np.transpose(sitk.GetArrayFromImage(reader.Execute()))[..., 0]
            if view != "sax":
                image = image[..., 0]
            data[view] = torch.from_numpy(image[None, ...])

        return self.transform(data)


def get_dataloader(config: DictConfig) -> DataLoader:
    """Get the dataloaders."""
    data_dir = Path(__file__).parent.parent.resolve() / "data" / "ukb"
    sax_paths = list(data_dir.glob("**/*_sax.nii.gz"))
    transform = Compose(
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
    dataset = UKBDataset(data=sax_paths, transform=transform)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=config.train.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=config.train.n_workers,
    )
    return dataloader


def pretrain_one_epoch(
    model: CineMA,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_scaler: GradScaler,
    device: torch.device,
    amp_dtype: torch.dtype,
    config: DictConfig,
    epoch: int,
) -> None:
    """Train one epoch.

    Args:
        model: model.
        dataloader: dataloader.
        optimizer: optimizer.
        loss_scaler: loss scaler.
        device: device.
        amp_dtype: dtype for automatic mixed precision.
        config: config.
        epoch: current epoch.
    """
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

        grad_norm = loss_scaler(
            loss=loss,
            optimizer=optimizer,
            clip_grad=clip_grad,
            parameters=model.parameters(),
            update_grad=True,
        )
        optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        metrics.update(
            {
                "grad_norm": grad_norm.item(),
                "lr": lr,
            },
        )
        logger.info(f"Metrics: {metrics}.")


@hydra.main(version_base=None, config_path="", config_name="pretrain")
def run(config: DictConfig) -> None:
    """Launch pre-training."""
    amp_dtype, device = get_amp_dtype_and_device()
    torch.manual_seed(config.seed)
    ckpt_dir = Path(config.logging.dir) / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataloader = get_dataloader(config)
    model = get_model(config)
    model.to(device)
    param_groups = param_groups_weight_decay(model=model, weight_decay=config.train.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.train.lr, betas=config.train.betas)
    loss_scaler = GradScaler()

    model.train(True)
    for epoch in range(config.train.n_epochs):
        optimizer.zero_grad()
        pretrain_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            device=device,
            amp_dtype=amp_dtype,
            config=config,
            epoch=epoch,
        )

        ckpt_path = ckpt_dir / f"ckpt_{epoch}.safetensors"
        save_file(model.state_dict(), ckpt_path)
        logger.info(f"Saved checkpoint of epoch {epoch} at {ckpt_path}.")


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
