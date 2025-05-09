"""Script to evaluate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from skimage.measure import label as LAB  # noqa: N812
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from cinema.data.myops2020 import MYOPS2020_LABEL_MAP
from cinema.data.sitk import (
    save_image,
)
from cinema.device import get_amp_dtype_and_device
from cinema.log import get_logger
from cinema.segmentation.dataset import get_segmentation_transforms
from cinema.segmentation.myops2020.train import MYOPS2020Dataset
from cinema.segmentation.train import get_segmentation_model, segmentation_eval

if TYPE_CHECKING:
    from omegaconf import DictConfig
logger = get_logger(__name__)


def postprocess_scar(
    label: np.ndarray,
    scar_class: int = 2221,
    threshold: int = 60,
) -> np.ndarray:
    """Postprocess the scar region.

    https://github.com/jianpengz/EfficientSeg/blob/main/post-processing.py

    Args:
        label: of shape (x, y, z).
        scar_class: class label for scar.
        threshold: threshold for removing small regions.

    Returns:
        post-processed label.
    """
    label_pp = label.copy()
    for i in range(label_pp.shape[2]):
        label_i = label_pp[:, :, i]
        regions = np.where(label_i == scar_class, np.ones_like(label_i), np.zeros_like(label_i))
        l_i, n_i = LAB(regions, background=0, connectivity=2, return_num=True)

        for j in range(1, n_i + 1):
            num_j = np.sum(l_i == j)
            if num_j < threshold:
                bbx_h, bbx_w = np.where(l_i == j)
                bbx_h_min = bbx_h.min()
                bbx_h_max = bbx_h.max()
                bbx_w_min = bbx_w.min()
                bbx_w_max = bbx_w.max()
                roi = label_i[bbx_h_min - 1 : bbx_h_max + 2, bbx_w_min - 1 : bbx_w_max + 2]
                replace_label = np.argmax(np.bincount(roi[roi != scar_class].flatten()))

                label_pp[:, :, i] = np.where(l_i == j, replace_label * np.ones_like(label_i), label_i)

    return label_pp


def postprocess_scar_and_edema(
    label: np.ndarray,
    edema_class: int = 1220,
    threshold: int = 200,
) -> np.ndarray:
    """Postprocess the scar region.

    https://github.com/jianpengz/EfficientSeg/blob/main/post-processing.py

    Args:
        label: of shape (x, y, z).
        edema_class: class label for edema.
        threshold: threshold for removing small regions.

    Returns:
        post-processed label.
    """
    label_pp = label.copy()
    for i in range(label_pp.shape[2]):
        label_i = label_pp[:, :, i]
        regions = np.where(label_i >= edema_class, np.ones_like(label_i), np.zeros_like(label_i))
        l_i, n_i = LAB(regions, background=0, connectivity=2, return_num=True)

        for j in range(1, n_i + 1):
            num_j = np.sum(l_i == j)
            if num_j < threshold:
                bbx_h, bbx_w = np.where(l_i == j)
                bbx_h_min = bbx_h.min()
                bbx_h_max = bbx_h.max()
                bbx_w_min = bbx_w.min()
                bbx_w_max = bbx_w.max()
                roi = label_i[bbx_h_min - 1 : bbx_h_max + 2, bbx_w_min - 1 : bbx_w_max + 2]
                replace_label = np.argmax(np.bincount(roi[roi < edema_class].flatten()))

                label_pp[:, :, i] = np.where(l_i == j, replace_label * np.ones_like(label_i), label_i)

    return label_pp


def segmentation_eval_myops2020_dataset(  # pylint:disable=too-many-statements
    config: DictConfig,
    split: str,
    ckpt_path: Path,
    out_dir: Path,
    save: bool,
) -> None:
    """Function to evaluate segmentation model on Myops2020 dataset.

    Args:
        config: config for evaluation.
        split: split of data, train or test.
        ckpt_path: path to the checkpoint.
        out_dir: output directory.
        save: whether to save the predictions.
    """
    if split != "test":
        raise ValueError("Only support test split for now.")
    out_dir = out_dir / split
    out_dir.mkdir(exist_ok=True, parents=True)
    # load data
    data_dir = Path(config.data.dir).expanduser()
    meta_df = pd.read_csv(data_dir / f"{split}_metadata.csv", dtype={"pid": str})
    if config.data.max_n_samples > 0:
        meta_df = meta_df.sample(n=min(config.data.max_n_samples, len(meta_df)), random_state=0)
        logger.info(f"Using {config.data.max_n_samples} samples for {split} split.")
    if config.model.views != "sax":
        raise ValueError("Only support sax view for now.")
    patch_size_dict = {"sax": config.data.sax.patch_size}
    spacing_dict = {"sax": config.data.sax.spacing}
    _, transform = get_segmentation_transforms(config)
    dataset = MYOPS2020Dataset(
        data_dir=data_dir / split,
        meta_df=meta_df,
        transform=transform,
    )
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=1,
        drop_last=False,
        pin_memory=True,
        num_workers=config.train.n_workers,
    )

    # load model
    amp_dtype, device = get_amp_dtype_and_device()
    model = get_segmentation_model(config)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # inference
    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):  # batch size is 1
        logits_dict, _ = segmentation_eval(
            model=model,
            batch=batch,
            patch_size_dict=patch_size_dict,
            spacing_dict=spacing_dict,
            amp_dtype=amp_dtype,
            device=device,
            metrics_fn=None,
        )

        # save segmentation
        if not save:
            continue

        # get segmentation mask
        width = batch["sax_width"][0]
        height = batch["sax_height"][0]
        n_slices = batch["n_slices"][0]
        pid = batch["pid"][0]
        logits = logits_dict["sax"]
        pred_label = torch.argmax(logits, dim=1).cpu().to(dtype=torch.int).numpy()  # (1, *image_size)
        pred_label = pred_label[0, :width, :height, :n_slices]  # (x, y, z)

        # pad to original size
        pad_width = (
            (int(batch["crop_lower_x"][0]), int(batch["crop_upper_x"][0])),
            (int(batch["crop_lower_y"][0]), int(batch["crop_upper_y"][0])),
            (0, 0),
        )
        pred_label = np.pad(pred_label, pad_width)

        # convert to original class labels
        for orig_cls, curr_cls in MYOPS2020_LABEL_MAP.items():
            pred_label[pred_label == curr_cls] = orig_cls

        # post processing
        # https://link.springer.com/chapter/10.1007/978-3-030-65651-5_2
        # For the predicted regions of scar, we remove the small regions with less than 60 voxels
        # and replace them with the surrounding labels.
        pred_label = postprocess_scar(pred_label)
        # For the predicted regions of edema and scars, we remove the small regions with less than 200 voxels
        # and replace them with the surrounding labels.
        # https://github.com/jianpengz/EfficientSeg/blob/main/post-processing.py
        pred_label = postprocess_scar_and_edema(pred_label)

        # save final segmentation
        true_image_path = data_dir.parent / "test20" / f"myops_test_{pid}_C0.nii.gz"
        pred_label_path = out_dir / f"{pid}_pred.nii.gz"
        save_image(
            image_np=pred_label,
            reference_image_path=true_image_path,
            out_path=pred_label_path,
        )


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to the checkpoint .pt file, config is assumed to be saved in the same directory.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Data directory, if not provided, using the one saved in config.",
        default=None,
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test"],
        help="Split of data test.",
        default="test",
    )
    parser.add_argument("--save", action="store_true", help="Save the predictions.")
    parser.set_defaults(save=False)
    args = parser.parse_args()

    return args


def main() -> None:
    """Main function."""
    args = parse_args()

    out_dir = args.ckpt_path.parent / f"myops2020_eval_{args.ckpt_path.stem}"
    out_dir.mkdir(exist_ok=True)

    # load and overwrite config
    config = OmegaConf.load(args.ckpt_path.parent / "config.yaml")
    if args.data_dir is not None:
        config.data.dir = Path(args.data_dir).expanduser()

    segmentation_eval_myops2020_dataset(
        config=config,
        split=args.split,
        ckpt_path=args.ckpt_path,
        out_dir=out_dir,
        save=args.save,
    )


if __name__ == "__main__":
    main()
