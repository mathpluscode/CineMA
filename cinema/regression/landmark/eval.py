"""Script to evaluate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from cinema.classification.train import get_classification_or_regression_model
from cinema.device import get_amp_dtype_and_device
from cinema.log import get_logger
from cinema.regression.landmark.train import get_coords_from_batch
from cinema.regression.train import regression_eval
from cinema.segmentation.landmark.dataset import LandmarkDetectionDataset
from cinema.segmentation.landmark.eval import draw_landmarks
from cinema.segmentation.landmark.train import landmark_detection_coords_metrics
from cinema.transform import crop_start

if TYPE_CHECKING:
    from omegaconf import DictConfig
logger = get_logger(__name__)


def landmark_regression_eval_dataset(  # pylint:disable=too-many-statements
    config: DictConfig,
    split: str,
    ckpt_path: Path,
    out_dir: Path,
    save: bool,
) -> None:
    """Function to evaluate segmentation model on end-diastole and end-systole dataset.

    Args:
        config: config for evaluation.
        split: train or val or test.
        ckpt_path: path to the checkpoint.
        out_dir: output directory.
        save: whether to save the predictions.
    """
    out_dir = out_dir / split
    out_dir.mkdir(exist_ok=True, parents=True)

    # load data
    data_dir = Path(config.data.dir).expanduser()
    view = config.model.views
    if not isinstance(view, str):
        raise TypeError(f"Multiple views not supported: {view}")
    meta_df = pd.read_csv(data_dir / f"{view}_{split}.csv")
    if config.data.max_n_samples > 0:
        meta_df = meta_df.sample(n=min(config.data.max_n_samples, len(meta_df)), random_state=0)
        logger.info(f"Using {config.data.max_n_samples} samples for {split} split.")
    transform = Compose(
        [
            ScaleIntensityd(keys=f"{view}_image"),
            SpatialPadd(
                keys=(f"{view}_image", f"{view}_label"),
                spatial_size=config.data.lax.patch_size,
                method="end",
                lazy=True,
            ),
        ]
    )
    dataset = LandmarkDetectionDataset(data_dir=data_dir, view=view, meta_df=meta_df, transform=transform)
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
    model = get_classification_or_regression_model(config)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)

    # inference
    patch_size_dict = {view: config.data.lax.patch_size}
    spacing_dict = {view: config.data.lax.spacing}
    metrics_list = []  # each element is a dict, corresponding to one sample
    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):  # batch size is 1
        # x1, y1, x2, y2, x3, y3
        true_labels, label_scales = get_coords_from_batch(batch, view)
        true_labels, label_scales = true_labels.to(device), label_scales.to(device)
        pred_labels, _ = regression_eval(
            model=model, batch=batch, patch_size_dict=patch_size_dict, amp_dtype=amp_dtype, device=device
        )
        pred_labels *= label_scales
        true_labels *= label_scales

        metrics = landmark_detection_coords_metrics(
            true_labels=true_labels,
            pred_labels=pred_labels,
            spacing=spacing_dict[view],
        )
        metrics = {k: v.item() for k, v in metrics.items()}

        # save metrics
        uid = batch["uid"][0]
        metrics["uid"] = uid
        metrics_list.append(metrics)

        # save heatmap
        if not save:
            continue

        # draw landmarks
        image_out_dir = out_dir / "image"
        image_out_dir.mkdir(exist_ok=True, parents=True)
        width = int(batch[f"{view}_width"][0])
        height = int(batch[f"{view}_height"][0])
        image = crop_start(batch[f"{view}_image"][0, 0].astype(torch.float32).cpu().numpy(), (width, height))
        draw_landmarks(image, metrics, [1, 2, 3], image_out_dir / f"{uid}.png")

    # save metrics per sample
    metric_df = pd.DataFrame(metrics_list).sort_values("uid")
    metric_path = out_dir / "metrics.csv"
    metric_df.to_csv(metric_path, index=False)
    logger.info(f"Saved metrics to {metric_path}.")

    metric_path = out_dir / "mean_metrics.csv"
    mean_metrics = {f"{k}_mean": v for k, v in metric_df.drop(columns=["uid"]).mean().to_dict().items()}
    std_metrics = {f"{k}_std": v for k, v in metric_df.drop(columns=["uid"]).std().to_dict().items()}
    pd.DataFrame([{**mean_metrics, **std_metrics}]).T.to_csv(metric_path, header=False)
    logger.info(f"Saved mean metrics to {metric_path}.")


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
        choices=["train", "test"],
        help="Split of data, train or test.",
        default="test",
    )
    parser.add_argument("--save", action="store_true", help="Save the predictions.")
    parser.set_defaults(save=False)
    args = parser.parse_args()

    return args


def main() -> None:
    """Main function."""
    args = parse_args()

    out_dir = args.ckpt_path.parent / f"landmark_detection_eval_{args.ckpt_path.stem}"
    out_dir.mkdir(exist_ok=True)

    # load and overwrite config
    config = OmegaConf.load(args.ckpt_path.parent / "config.yaml")
    if args.data_dir is not None:
        config.data.dir = Path(args.data_dir).expanduser()

    landmark_regression_eval_dataset(
        config=config,
        split=args.split,
        ckpt_path=args.ckpt_path,
        out_dir=out_dir,
        save=args.save,
    )


if __name__ == "__main__":
    main()
