"""Script to evaluate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from cinema.device import get_amp_dtype_and_device
from cinema.log import get_logger
from cinema.segmentation.landmark.dataset import LandmarkDetectionDataset
from cinema.segmentation.landmark.train import landmark_detection_eval
from cinema.segmentation.train import get_segmentation_model
from cinema.transform import crop_start

if TYPE_CHECKING:
    from omegaconf import DictConfig
logger = get_logger(__name__)


def draw_landmarks(image: np.ndarray, metrics: dict[str, float], landmarks: list[int], out_path: Path) -> None:
    """Draw landmarks on the image.

    Args:
        image: (width, height) image or heatmap.
        metrics: metrics having predicted and ground truth landmarks.
        landmarks: list of landmarks to draw.
        out_path: output path.
    """
    image = image[..., None] * np.array([255, 255, 255])[None, None, :]
    image = image.clip(0, 255).astype(np.uint8)
    for i in landmarks:
        # draw predictions with blue cross
        pred_x, pred_y = metrics[f"pred_x{i}"], metrics[f"pred_y{i}"]
        pred_x, pred_y = int(pred_x), int(pred_y)
        x1, x2 = max(0, pred_x - 7), min(image.shape[0], pred_x + 8)
        y1, y2 = max(0, pred_y - 7), min(image.shape[1], pred_y + 8)
        if pred_y < 0 or pred_y >= image.shape[1] or pred_x < 0 or pred_x >= image.shape[0]:
            logger.error(f"Predicted landmark {i} out of bounds: ({pred_x}, {pred_y}), for image shape {image.shape}.")

        image[pred_x, y1:y2] = [0, 0, 255]
        image[x1:x2, pred_y] = [0, 0, 255]
        # draw ground truth with red cross, slightly smaller to not overlap with pred
        true_x, true_y = metrics[f"true_x{i}"], metrics[f"true_y{i}"]
        true_x, true_y = int(true_x), int(true_y)
        x1, x2 = max(0, true_x - 5), min(image.shape[0], true_x + 6)
        y1, y2 = max(0, true_y - 5), min(image.shape[1], true_y + 6)
        image[true_x, y1:y2] = [255, 0, 0]
        image[x1:x2, true_y] = [255, 0, 0]
    out_path.parent.mkdir(exist_ok=True, parents=True)
    Image.fromarray(np.moveaxis(image, 0, 1)).save(out_path)


def landmark_detection_eval_dataset(  # pylint:disable=too-many-statements
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
    data_dir = Path(config.data.dir)
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
    model = get_segmentation_model(config)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # inference
    patch_size_dict = {view: config.data.lax.patch_size}
    spacing_dict = {view: config.data.lax.spacing}
    metrics_list = []  # each element is a dict, corresponding to one sample
    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):  # batch size is 1
        logits_dict, metrics = landmark_detection_eval(
            model=model,
            batch=batch,
            patch_size_dict=patch_size_dict,
            spacing_dict=spacing_dict,
            amp_dtype=amp_dtype,
            device=device,
        )

        # save metrics
        uid = batch["uid"][0]
        metrics["uid"] = uid
        metrics_list.append(metrics)

        # save heatmap
        if not save:
            continue

        # save final segmentation
        npz_out_dir = out_dir / "npz"
        npz_out_dir.mkdir(exist_ok=True, parents=True)
        logits = logits_dict[view]  # (3, width, height)
        probs = torch.sigmoid(logits)[0].astype(torch.float32).cpu().numpy()  # (3, width, height)
        np.savez_compressed(
            npz_out_dir / f"{uid}.npz", probs=probs, logits=logits[0].astype(torch.float32).cpu().numpy()
        )

        heatmap_out_dir = out_dir / "heatmap"
        heatmap_out_dir.mkdir(exist_ok=True, parents=True)
        draw_landmarks(probs[0], metrics, [1], heatmap_out_dir / f"{uid}_landmark1.png")
        draw_landmarks(probs[1], metrics, [2], heatmap_out_dir / f"{uid}_landmark2.png")
        draw_landmarks(probs[2], metrics, [3], heatmap_out_dir / f"{uid}_landmark3.png")

        image_out_dir = out_dir / "image"
        image_out_dir.mkdir(exist_ok=True, parents=True)
        image = crop_start(batch[f"{view}_image"][0, 0].astype(torch.float32).cpu().numpy(), probs.shape[-2:])
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

    landmark_detection_eval_dataset(
        config=config,
        split=args.split,
        ckpt_path=args.ckpt_path,
        out_dir=out_dir,
        save=args.save,
    )


if __name__ == "__main__":
    main()
