"""A minimalist script to train a segmentation model on ACDC dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import pandas as pd
import SimpleITK as sitk  # noqa: N813
import torch
from huggingface_hub import snapshot_download
from monai.losses import DiceLoss
from monai.metrics import compute_dice, compute_hausdorff_distance
from monai.networks.utils import one_hot
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandAffined,
    RandCoarseDropoutd,
    RandGaussianNoised,
    RandSpatialCropd,
    ScaleIntensityd,
    SpatialPadd,
    Transform,
)
from safetensors.torch import save_file
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from cinema import ConvUNetR
from cinema.convvit import param_groups_lr_decay
from cinema.device import get_amp_dtype_and_device
from cinema.log import get_logger
from cinema.optim import EarlyStopping, GradScaler, adjust_learning_rate

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = get_logger(__name__)


class ACDCDataset(Dataset):
    """Dataset for ACDC ED/ES frame segmentation."""

    def __init__(
        self,
        data_dir: Path,
        meta_df: pd.DataFrame,
        transform: Transform,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize the dataset."""
        self.data_dir = data_dir
        self.meta_df = meta_df
        self.transform = transform
        self.dtype = dtype

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.meta_df) * 2

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a sample."""
        row_idx, is_ed = idx // 2, idx % 2 == 0
        row = self.meta_df.iloc[row_idx]
        pid = row["pid"]
        pid_dir = self.data_dir / str(pid)

        frame_name = "ed" if is_ed else "es"
        image_path = pid_dir / f"{pid}_sax_{frame_name}.nii.gz"
        label_path = pid_dir / f"{pid}_sax_{frame_name}_gt.nii.gz"
        image = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(image_path)))  # (x, y, z)
        label = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(label_path)))  # same shape as image
        data = {
            "pid": pid,
            "is_ed": is_ed,
            "sax_width": image.shape[0],
            "sax_height": image.shape[1],
            "n_slices": int(row["n_slices"]),
            "sax_image": torch.from_numpy(image[None, ...]),  # (1, x, y, z)
            "sax_label": torch.from_numpy(label[None, ...].astype(np.int8)),
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
    dropout_size_dict = {"sax": config.transform.sax.dropout_size}
    train_transform = Compose(
        [
            RandAdjustContrastd(keys="sax_image", prob=config.transform.prob, gamma=config.transform.gamma),
            RandGaussianNoised(keys="sax_image", prob=config.transform.prob),
            ScaleIntensityd(keys="sax_image"),
            RandAffined(
                keys=("sax_image", "sax_label"),
                mode=("bilinear", "nearest"),
                prob=config.transform.prob,
                rotate_range=tuple(r / 180 * np.pi for r in rotate_range_dict["sax"]),
                translate_range=translate_range_dict["sax"],
                scale_range=config.transform.scale_range,
                padding_mode="zeros",
                lazy=True,
            ),
            RandCoarseDropoutd(
                keys="sax_image",
                prob=config.transform.prob,
                holes=1,
                fill_value=0,
                spatial_size=dropout_size_dict["sax"],
            ),
            RandSpatialCropd(
                keys=("sax_image", "sax_label"),
                roi_size=patch_size_dict["sax"],
                lazy=True,
            ),
            SpatialPadd(
                keys=("sax_image", "sax_label"),
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
                keys=("sax_image", "sax_label"),
                spatial_size=patch_size_dict["sax"],
                method="end",
                lazy=True,
            ),
        ]
    )
    train_dataset = ACDCDataset(data_dir=data_dir / "train", meta_df=train_meta_df, transform=train_transform)
    val_dataset = ACDCDataset(data_dir=data_dir / "train", meta_df=val_meta_df, transform=val_transform)

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


@hydra.main(version_base=None, config_path="", config_name="segmentation")
def run(config: DictConfig) -> None:
    """Entrypoint for segmentation training.

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
    model = ConvUNetR.from_pretrained(
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
    spacing = config.data.sax.spacing
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
                logits = model({"sax": batch["sax_image"].to(device)})["sax"]
                labels = batch["sax_label"].to(device).long()
                mask = one_hot(labels.clamp(min=0), num_classes=logits.shape[1], dtype=logits.dtype)
                ce = F.cross_entropy(logits, labels.squeeze(dim=1), ignore_index=-1)
                dice = DiceLoss(include_background=False, to_onehot_y=False, softmax=True)(logits, mask)
                loss = dice + ce
                metrics = {
                    "train_cross_entropy": ce.item(),
                    "train_mean_dice_loss": dice.item(),
                    "train_loss": loss.item(),
                }

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
        dices = []
        hds = []
        for _, batch in enumerate(val_dataloader):
            with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                logits = model({"sax": batch["sax_image"].to(device)})["sax"]
                true_labels = batch["sax_label"].squeeze(dim=1).long()  # (x, y, z)
                pred_labels = torch.argmax(logits, dim=1).cpu()
                pred_mask = F.one_hot(pred_labels, 4).moveaxis(-1, 1)
                true_mask = F.one_hot(true_labels, 4).moveaxis(-1, 1)
                dices.append(
                    compute_dice(
                        y_pred=pred_mask,
                        y=true_mask,
                        num_classes=4,
                    )
                )
                hds.append(
                    compute_hausdorff_distance(
                        y_pred=pred_mask,
                        y=true_mask,
                        percentile=95,
                        spacing=spacing,
                    )
                )
        metrics = {"val_mean_dice_score": np.mean(dices), "val_mean_hausdorff_distance": np.mean(hds)}
        metrics = {k: f"{v:.2e}" for k, v in metrics.items()}
        logger.info(f"Validation metrics: {metrics}.")

        # save model checkpoint
        ckpt_path = ckpt_dir / f"ckpt_{epoch}.safetensors"
        save_file(model.state_dict(), ckpt_path)
        logger.info(f"Saved checkpoint of epoch {epoch} at {ckpt_path} after {n_samples} samples.")

        # early stopping
        early_stop.update(-np.mean(dices))
        if early_stop.should_stop:
            logger.info(
                f"Met early stopping criteria with {config.train.early_stopping.metric} = "
                f"{early_stop.best_metric} and patience {early_stop.patience_count}, breaking."
            )
            break


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
