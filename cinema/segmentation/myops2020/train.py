"""Script to train."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import pandas as pd
import SimpleITK as sitk  # noqa: N813
import torch
from monai.metrics import compute_dice, compute_hausdorff_distance, compute_iou
from torch.utils.data import Dataset

from cinema.log import get_logger
from cinema.metric import get_volumes
from cinema.segmentation.dataset import get_segmentation_transforms
from cinema.segmentation.train import (
    get_segmentation_model,
    segmentation_eval_dataloader,
    segmentation_loss,
)
from cinema.train import maybe_subset_dataset, run_train

if TYPE_CHECKING:
    from monai.transforms import Transform
    from omegaconf import DictConfig

logger = get_logger(__name__)


class MYOPS2020Dataset(Dataset):
    """Dataset for MYOPS2020.

    Images are stored in 3d, of size (x, y, z).

    ```
    data_dir/
    ├── pid1/
    │   ├── pid1_c0.nii.gz
    │   ├── pid1_de.nii.gz
    │   ├── pid1_t2.nii.gz
    │   └── pid1_gt.nii.gz
    ├── pid2/
    ├── .../
    ```
    """

    def __init__(
        self,
        data_dir: Path,
        meta_df: pd.DataFrame,
        transform: Transform | None = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_dir: path to the data directory.
            meta_df: metadata dataframe, including pid and n_slices.
            transform: data augmentation to be applied per sample.
            dtype: data type for image.
        """
        for col in ["pid", "n_slices"]:
            if col not in meta_df.columns:
                raise ValueError(f"Column {col} is required in meta_df.")
        self.data_dir = data_dir
        self.meta_df = meta_df
        self.views = ["sax"]
        self.transform = transform
        self.dtype = dtype

    def __len__(self) -> int:
        """Return the number of samples.

        Returns:
            Number of samples.
        """
        return len(self.meta_df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a sample.

        Args:
            idx: index of the sample.

        Returns:
            A dictionary having ED/ES image and label.
            Particularly, label is of type int8, not uint8.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.meta_df.iloc[idx]
        pid = str(int(row["pid"]))
        pid_dir = self.data_dir / pid

        image_c0_path = pid_dir / f"{pid}_c0.nii.gz"  # bssfp
        image_de_path = pid_dir / f"{pid}_de.nii.gz"  # LGE
        image_t2_path = pid_dir / f"{pid}_t2.nii.gz"  # T2

        image_c0 = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(image_c0_path)))  # (x, y, z)
        image_de = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(image_de_path)))  # (x, y, z)
        image_t2 = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(image_t2_path)))  # (x, y, z)

        image = np.stack([image_c0, image_de, image_t2], axis=0)  # (3, x, y, z)

        data = {
            "pid": pid,
            "orig_spacing_x": float(row["orig_spacing_x"]),
            "orig_spacing_y": float(row["orig_spacing_y"]),
            "orig_spacing_z": float(row["orig_spacing_z"]),
            "crop_lower_x": int(row["crop_lower_x"]),
            "crop_lower_y": int(row["crop_lower_y"]),
            "crop_upper_x": int(row["crop_upper_x"]),
            "crop_upper_y": int(row["crop_upper_y"]),
            "n_slices": int(row["n_slices"]),
        }
        data["sax_width"], data["sax_height"] = image_c0.shape[:2]
        data["sax_image"] = torch.from_numpy(image)  #  (3, x, y, z)

        label_path = pid_dir / f"{pid}_gt.nii.gz"
        if label_path.exists():
            label_np = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(label_path)))  # (x, y, z)
            data["sax_label"] = torch.from_numpy(label_np[None, ...].astype(np.int8))

        if self.transform:
            data = self.transform(data)
        return data


def load_dataset(config: DictConfig) -> tuple[Dataset, Dataset]:
    """Load and split the dataset.

    Args:
        config: config for fine-tuning.

    Returns:
        train_dataset: dataset for training.
        val_dataset: dataset for validation.
    """
    data_dir = Path(config.data.dir)
    meta_df = pd.read_csv(data_dir / "train_metadata.csv")
    val_pids = meta_df.sample(n=5, random_state=0)["pid"].tolist()
    train_meta_df = meta_df[~meta_df["pid"].isin(val_pids)].reset_index(drop=True)
    val_meta_df = meta_df[meta_df["pid"].isin(val_pids)].reset_index(drop=True)
    train_meta_df, val_meta_df = maybe_subset_dataset(config, train_meta_df, val_meta_df)

    train_transform, val_transform = get_segmentation_transforms(config)
    train_dataset = MYOPS2020Dataset(data_dir=data_dir / "train", meta_df=train_meta_df, transform=train_transform)
    val_dataset = MYOPS2020Dataset(data_dir=data_dir / "train", meta_df=val_meta_df, transform=val_transform)
    return train_dataset, val_dataset


def myops2020_segmentation_eval_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    spacing: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    """Evaluation metrics for segmentation tasks.

    Padded region is considered as background as the metrics are focused on foreground classes.
    Logits are modified to have background class at index 0.

    Args:
        logits: (batch, 6, ...), n_classes is number of foreground classes.
        labels: target labels, (batch, 1, ...).
        spacing: pixel/voxel spacing in mm.

    Returns:
        metrics: each value is of shape (batch,).
    """
    dtype = logits.dtype
    metrics = {}
    labels = labels.long()
    pred_labels = torch.argmax(logits, dim=1, keepdim=True).long()  # (batch, ...)

    true_mask = torch.concat(
        [
            labels == 0,  # background
            labels == 1,  # RV
            labels == 2,  # LV
            labels >= 3,  # LV Myocardium
            labels >= 4,  # LV myocardial edema
            labels == 5,  # LV myocardial scar
        ],
        dim=1,
    ).to(dtype)  # (batch, 6, ...)
    pred_mask = torch.concat(
        [
            pred_labels == 0,
            pred_labels == 1,
            pred_labels == 2,
            pred_labels >= 3,
            pred_labels >= 4,
            pred_labels == 5,
        ],
        dim=1,
    ).to(dtype)  # (batch, 6, ...)

    dice = compute_dice(
        y_pred=pred_mask,
        y=true_mask,
        # https://github.com/EMIDEC-Challenge/Evaluation-metrics/blob/master/metrics.py#L73
        # when ignore_empty is False, the dice score is 1 if both true and pred are empty
        ignore_empty=False,
    )  # (batch, 5)
    iou = compute_iou(
        y_pred=pred_mask,
        y=true_mask,
    )  # (batch, 5)
    hausdorff_dist = compute_hausdorff_distance(
        y_pred=pred_mask,
        y=true_mask,
        percentile=95,
        spacing=spacing,
    )  # (batch, 4)
    true_volumes = get_volumes(mask=true_mask, spacing=spacing)  # (batch, 5)
    pred_volumes = get_volumes(mask=pred_mask, spacing=spacing)  # (batch, 5)

    for cls_idx in range(1, 6):
        metrics[f"class_{cls_idx}_dice_score"] = dice[:, cls_idx]
        metrics[f"class_{cls_idx}_iou_score"] = iou[:, cls_idx]
        metrics[f"class_{cls_idx}_hausdorff_distance_95"] = hausdorff_dist[:, cls_idx - 1]
        metrics[f"class_{cls_idx}_true_volume"] = true_volumes[:, cls_idx]
        metrics[f"class_{cls_idx}_pred_volume"] = pred_volumes[:, cls_idx]

    # not all classes are present in all samples
    metrics["mean_dice_score"] = torch.nanmean(dice[:, 1:], dim=-1)
    metrics["mean_iou_score"] = torch.nanmean(iou[:, 1:], dim=-1)
    metrics["mean_hausdorff_distance_95"] = torch.nanmean(hausdorff_dist, dim=-1)

    return metrics


myops2020_segmentation_eval_dataloader = partial(
    segmentation_eval_dataloader, metrics_fn=myops2020_segmentation_eval_metrics
)


@hydra.main(version_base=None, config_path="", config_name="config")
def main(config: DictConfig) -> None:
    """Entrypoint for segmentation training.

    Args:
        config: config loaded from yaml.
    """
    run_train(
        config=config,
        load_dataset=load_dataset,
        get_model_fn=get_segmentation_model,
        loss_fn=segmentation_loss,
        eval_dataloader_fn=myops2020_segmentation_eval_dataloader,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
