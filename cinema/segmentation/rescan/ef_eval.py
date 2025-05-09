"""Script to evaluate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from cinema.data.sitk import save_image
from cinema.device import get_amp_dtype_and_device
from cinema.log import get_logger
from cinema.metric import get_volumes
from cinema.segmentation.dataset import get_segmentation_transforms
from cinema.segmentation.rescan.dataset import CineSegmentationDataset
from cinema.segmentation.train import get_segmentation_model, segmentation_eval

if TYPE_CHECKING:
    from omegaconf import DictConfig
logger = get_logger(__name__)


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
        choices=["test_retest_100"],
        help="Split of data, test_retest_100.",
        default="test_retest_100",
    )
    parser.add_argument("--save", action="store_true", help="Save the predictions.")
    parser.set_defaults(save=False)
    args = parser.parse_args()

    return args


def ef_metrics(
    logits: torch.Tensor,
    spacing: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    """Evaluation metrics for segmentation tasks.

    Args:
        logits: (batch, 1+n_classes, ...), n_classes is number of foreground classes.
        spacing: pixel/voxel spacing in mm.

    Returns:
        metrics: each value is of shape (batch,).
    """
    n_classes = logits.shape[1] - 1  # exclude background class
    pred_labels = torch.argmax(logits, dim=1)
    pred_mask = F.one_hot(pred_labels, n_classes + 1).moveaxis(-1, 1)
    pred_volumes = get_volumes(mask=pred_mask, spacing=spacing)  # (batch, n_classes+1)

    metrics = {}
    for i in range(n_classes):
        metrics[f"class_{i + 1}_pred_volume"] = pred_volumes[:, i + 1]

    return metrics


def ef_eval_rescan_dataset(
    config: DictConfig,
    split: str,
    ckpt_path: Path,
    out_dir: Path,
    save: bool,
) -> None:
    """Function to evaluate model dataset.

    Args:
        config: config for evaluation.
        split: split of data, train or test.
        ckpt_path: path to the checkpoint.
        out_dir: output directory.
        save: whether to save the predictions.
    """
    # load data
    data_dir = Path(config.data.dir).expanduser()
    meta_df = pd.read_csv(data_dir / f"{split}_metadata.csv", dtype={"pid": str})
    if config.data.max_n_samples > 0:
        meta_df = meta_df.sample(n=min(config.data.max_n_samples, len(meta_df)), random_state=0)
        logger.info(f"Using {config.data.max_n_samples} samples for {split} split.")
    views = [config.model.views] if isinstance(config.model.views, str) else config.model.views
    if len(views) != 1:
        raise TypeError("Only support one view for evaluation.")
    patch_size_dict = {
        view: config.data.sax.patch_size if view == "sax" else config.data.lax.patch_size for view in views
    }
    spacing_dict = {view: config.data.sax.spacing if view == "sax" else config.data.lax.spacing for view in views}
    _, transform = get_segmentation_transforms(config)
    dataset = CineSegmentationDataset(
        data_dir=data_dir / split,
        meta_df=meta_df,
        views=views,
        has_labels=False,
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
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # inference
    metrics_list = []
    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):  # batch size is 1
        logits_dict, _ = segmentation_eval(
            model=model,
            batch=batch,
            patch_size_dict=patch_size_dict,
            spacing_dict=spacing_dict,
            amp_dtype=amp_dtype,
            device=device,
            metrics_fn=None,  # calculate separately
        )
        metrics = {}
        for view in views:
            metrics_view = ef_metrics(logits_dict[view], spacing_dict[view])
            metric_keys = list(metrics_view.keys())
            for k, v in metrics_view.items():
                metrics[f"{view}_{k}"] = float(v.cpu().to(dtype=torch.float32).numpy())
        for k in metric_keys:
            metrics[k] = np.mean([metrics[f"{view}_{k}"] for view in views])

        # save metrics
        pid = batch["pid"][0]
        frame = int(batch["frame"].numpy())  # frame of the slice
        metrics["pid"] = pid
        metrics["n_slices"] = int(batch["n_slices"].numpy())
        metrics["edv"] = float(batch["edv"].numpy())  # same per frame
        metrics["esv"] = float(batch["esv"].numpy())
        metrics["ef"] = float(batch["ef"].numpy())
        metrics["frame"] = frame
        metrics_list.append(metrics)

        # save segmentation
        if not save:
            continue
        pred_labels_dict = {
            view: torch.argmax(logits_dict[view][0], dim=0).cpu().to(dtype=torch.uint8).numpy() for view in views
        }
        for view in views:
            true_image_path = data_dir / split / pid / f"{view}_t.nii.gz"
            pred_label_path = out_dir / split / pid / f"{view}_pred_t{frame}.nii.gz"
            image_np = pred_labels_dict[view] if view == "sax" else pred_labels_dict[view][..., None]
            save_image(
                image_np=image_np,
                reference_image_path=true_image_path,
                out_path=pred_label_path,
            )

    # save metrics per sample
    metric_path = out_dir / split / "metrics.csv"
    metric_path.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(metrics_list).sort_values("pid").to_csv(metric_path, index=False)
    logger.info(f"Saved metrics to {metric_path}.")


def main() -> None:
    """Main function."""
    args = parse_args()

    out_dir = args.ckpt_path.parent / f"rescan_eval_{args.ckpt_path.stem}"
    out_dir.mkdir(exist_ok=True)

    # load and overwrite config
    config = OmegaConf.load(args.ckpt_path.parent / "config.yaml")
    if args.data_dir is not None:
        config.data.dir = Path(args.data_dir).expanduser()

    ef_eval_rescan_dataset(
        config=config,
        split=args.split,
        ckpt_path=args.ckpt_path,
        out_dir=out_dir,
        save=args.save,
    )


if __name__ == "__main__":
    main()
