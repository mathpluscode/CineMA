"""A minimalist script to train a regression model on ACDC dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import pandas as pd
import SimpleITK as sitk  # noqa: N813
import torch
from huggingface_hub import snapshot_download
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandSpatialCropd,
    ScaleIntensityd,
    SpatialPadd,
    Transform,
)
from safetensors.torch import save_file
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from cinema import ConvViT
from cinema.convvit import param_groups_lr_decay
from cinema.device import get_amp_dtype_and_device
from cinema.log import get_logger
from cinema.optim import EarlyStopping, GradScaler, adjust_learning_rate

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = get_logger(__name__)


class ACDCDataset(Dataset):
    """Dataset for ACDC ED/ES frame regression."""

    def __init__(
        self,
        data_dir: Path,
        meta_df: pd.DataFrame,
        reg_col: str,
        reg_mean: float,
        reg_std: float,
        transform: Transform,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize the dataset."""
        self.data_dir = data_dir
        self.meta_df = meta_df
        self.transform = transform
        self.dtype = dtype
        self.reg_col = reg_col
        self.reg_mean = reg_mean
        self.reg_std = reg_std

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.meta_df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a sample."""
        row = self.meta_df.iloc[idx]
        pid = row["pid"]
        pid_dir = self.data_dir / str(pid)

        ed_image_path = pid_dir / f"{pid}_sax_ed.nii.gz"
        es_image_path = pid_dir / f"{pid}_sax_es.nii.gz"
        ed_image = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(ed_image_path)))  # (x, y, z)
        es_image = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(es_image_path)))
        data = {
            "pid": pid,
            "sax_image": torch.from_numpy(np.stack([ed_image, es_image], axis=0)),  # (2, x, y, z)
            "label": torch.tensor([(row[self.reg_col] - self.reg_mean) / self.reg_std], dtype=torch.float32),
        }
        data = self.transform(data)
        return data


def get_dataloaders(config: DictConfig) -> tuple[DataLoader, DataLoader]:
    """Get the dataloaders."""
    data_dir = Path(
        snapshot_download(repo_id="mathpluscode/ACDC", allow_patterns=["*.nii.gz", "*.csv"], repo_type="dataset")
    )
    meta_df = pd.read_csv(data_dir / "train.csv")

    val_pids = meta_df.groupby("pathology").sample(n=2, random_state=0)["pid"].tolist()
    train_meta_df = meta_df[~meta_df["pid"].isin(val_pids)].reset_index(drop=True)
    if config.data.max_n_samples > 0:
        train_meta_df = train_meta_df.head(config.data.max_n_samples)
        logger.warning(f"Using {len(train_meta_df)} samples instead of {config.data.max_n_samples}.")
    val_meta_df = meta_df[meta_df["pid"].isin(val_pids)].reset_index(drop=True)

    patch_size_dict = {"sax": config.data.sax.patch_size}
    rotate_range_dict = {"sax": config.transform.sax.rotate_range}
    translate_range_dict = {"sax": config.transform.sax.translate_range}
    reg_col = config.data.regression_column
    reg_mean = config.data[reg_col].mean
    reg_std = config.data[reg_col].std
    train_transform = Compose(
        [
            RandAdjustContrastd(keys="sax_image", prob=config.transform.prob, gamma=config.transform.gamma),
            RandGaussianNoised(keys="sax_image", prob=config.transform.prob),
            ScaleIntensityd(keys="sax_image"),
            RandAffined(
                keys="sax_image",
                mode="bilinear",
                prob=config.transform.prob,
                rotate_range=tuple(r / 180 * np.pi for r in rotate_range_dict["sax"]),
                translate_range=translate_range_dict["sax"],
                scale_range=config.transform.scale_range,
                padding_mode="zeros",
                lazy=True,
            ),
            RandSpatialCropd(
                keys="sax_image",
                roi_size=patch_size_dict["sax"],
                lazy=True,
            ),
            SpatialPadd(
                keys="sax_image",
                spatial_size=patch_size_dict["sax"],
                method="end",
                lazy=True,
            ),
        ]
    )
    val_transform = Compose(
        [
            ScaleIntensityd(keys="sax_image"),
            SpatialPadd(
                keys="sax_image",
                spatial_size=patch_size_dict["sax"],
                method="end",
                lazy=True,
            ),
        ]
    )
    train_dataset = ACDCDataset(
        data_dir=data_dir / "train",
        meta_df=train_meta_df,
        transform=train_transform,
        reg_col=reg_col,
        reg_mean=reg_mean,
        reg_std=reg_std,
    )
    val_dataset = ACDCDataset(
        data_dir=data_dir / "train",
        meta_df=val_meta_df,
        transform=val_transform,
        reg_col=reg_col,
        reg_mean=reg_mean,
        reg_std=reg_std,
    )

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=config.train.n_workers,
    )

    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=1,
        drop_last=False,
        pin_memory=True,
        num_workers=config.train.n_workers,
    )
    return train_dataloader, val_dataloader


@hydra.main(version_base=None, config_path="", config_name="regression")
def main(config: DictConfig) -> None:
    """Entrypoint for regression training.

    Args:
        config: config loaded from yaml.
    """
    amp_dtype, device = get_amp_dtype_and_device()
    torch.manual_seed(config.seed)
    ckpt_dir = Path(config.logging.dir) / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # load dataset
    train_dataloader, val_dataloader = get_dataloaders(config=config)

    # load model
    model = ConvViT.from_pretrained(
        config=config,
        freeze=config.model.freeze_pretrained,
    )
    param_groups = param_groups_lr_decay(
        model,
        no_weight_decay_list=[],
        weight_decay=config.train.weight_decay,
        layer_decay=config.train.layer_decay,
    )
    model.set_grad_ckpt(config.grad_ckpt)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"number of parameters: {n_params:,}")
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of trainable parameters: {n_trainable_params:,}")

    # init optimizer
    logger.info("Initializing optimizer.")
    optimizer = torch.optim.AdamW(param_groups, lr=config.train.lr, betas=config.train.betas)
    loss_scaler = GradScaler()

    # train
    logger.info("Start training.")
    early_stop = EarlyStopping(
        min_delta=config.train.early_stopping.min_delta,
        patience=config.train.early_stopping.patience,
    )
    n_samples = 0
    for epoch in range(config.train.n_epochs):
        optimizer.zero_grad()
        model.train()
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
                preds = model({"sax": batch["sax_image"].to(device)})
                label = batch["label"].to(dtype=preds.dtype, device=device)
                loss = F.mse_loss(preds, label)
                metrics = {"train_mse_loss": loss.item()}

            grad_norm = loss_scaler(
                loss=loss,
                optimizer=optimizer,
                clip_grad=config.train.clip_grad,
                parameters=model.parameters(),
                update_grad=True,
            )
            optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            n_samples += batch["sax_image"].shape[0]
            metrics.update({"grad_norm": grad_norm.item(), "lr": lr})
            metrics = {k: f"{v:.2e}" for k, v in metrics.items()}
            logger.info(f"Epoch {epoch}, step {i}, metrics: {metrics}.")

        if (ckpt_dir is None) or ((epoch + 1) % config.train.eval_interval != 0):
            continue

        # evaluate model
        logger.info(f"Start evaluating model at epoch {epoch}.")
        model.eval()
        errors = []
        for _, batch in enumerate(val_dataloader):
            with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                preds = model({"sax": batch["sax_image"].to(device)})
                labels = batch["label"].to(dtype=preds.dtype, device=device)
                errors.append(torch.abs(preds - labels).item() * val_dataloader.dataset.reg_std)
        metrics = {"val_mae": np.mean(errors)}
        metrics = {k: f"{v:.2e}" for k, v in metrics.items()}
        logger.info(f"Validation metrics: {metrics}.")

        # save model checkpoint
        ckpt_path = ckpt_dir / f"ckpt_{epoch}.safetensors"
        save_file(model.state_dict(), ckpt_path)
        logger.info(f"Saved checkpoint of epoch {epoch} at {ckpt_path} after {n_samples} samples.")

        # early stopping
        early_stop.update(np.mean(errors))
        if early_stop.should_stop:
            logger.info(
                f"Met early stopping criteria with {config.train.early_stopping.metric} = "
                f"{early_stop.best_metric} and patience {early_stop.patience_count}, breaking."
            )
            break


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
