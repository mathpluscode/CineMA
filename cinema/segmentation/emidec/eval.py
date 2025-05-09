"""Script to evaluate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from cinema.data.sitk import save_image
from cinema.device import get_amp_dtype_and_device
from cinema.log import get_logger
from cinema.segmentation.dataset import get_segmentation_transforms
from cinema.segmentation.emidec.train import EMIDECDataset, emidec_segmentation_eval_metrics
from cinema.segmentation.train import get_segmentation_model, segmentation_eval

if TYPE_CHECKING:
    from omegaconf import DictConfig
logger = get_logger(__name__)


def segmentation_eval_emidec_dataset(  # pylint:disable=too-many-statements
    config: DictConfig,
    split: str,
    ckpt_path: Path,
    out_dir: Path,
    save: bool,
) -> None:
    """Function to evaluate segmentation model on EMIDEC dataset.

    Args:
        config: config for evaluation.
        split: split of data, train or test.
        ckpt_path: path to the checkpoint.
        out_dir: output directory.
        save: whether to save the predictions.
    """
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
    dataset = EMIDECDataset(
        data_dir=data_dir / "train",
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
    metrics_list = []  # each element is a dict, corresponding to one sample
    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):  # batch size is 1
        logits_dict, metrics = segmentation_eval(
            model=model,
            batch=batch,
            patch_size_dict=patch_size_dict,
            spacing_dict=spacing_dict,
            amp_dtype=amp_dtype,
            device=device,
            metrics_fn=emidec_segmentation_eval_metrics,
        )

        # save metrics
        pid = batch["pid"][0]
        metrics["pid"] = pid
        metrics_list.append(metrics)

        # save segmentation
        if not save:
            continue

        # save final segmentation
        true_label_path = data_dir / "train" / pid / f"{pid}_gt.nii.gz"
        pred_label_path = out_dir / pid / "Contours" / f"{pid}.nii.gz"
        logits = logits_dict["sax"]
        pred_labels = torch.argmax(logits, dim=1)[0].astype(torch.float32).cpu().numpy()  # (x, y, z)
        save_image(
            image_np=pred_labels,
            reference_image_path=true_label_path,
            out_path=pred_label_path,
        )

    # save metrics per sample
    metric_df = pd.DataFrame(metrics_list).sort_values("pid")
    metric_path = out_dir / "metrics.csv"
    metric_df.to_csv(metric_path, index=False)
    logger.info(f"Saved metrics to {metric_path}.")

    metric_path = out_dir / "mean_metrics.csv"
    mean_metrics = {f"{k}_mean": v for k, v in metric_df.drop(columns=["pid"]).mean().to_dict().items()}
    std_metrics = {f"{k}_std": v for k, v in metric_df.drop(columns=["pid"]).std().to_dict().items()}
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

    out_dir = args.ckpt_path.parent / f"emidec_eval_{args.ckpt_path.stem}"
    out_dir.mkdir(exist_ok=True)

    # load and overwrite config
    config = OmegaConf.load(args.ckpt_path.parent / "config.yaml")
    if args.data_dir is not None:
        config.data.dir = Path(args.data_dir).expanduser()

    segmentation_eval_emidec_dataset(
        config=config,
        split=args.split,
        ckpt_path=args.ckpt_path,
        out_dir=out_dir,
        save=args.save,
    )


if __name__ == "__main__":
    main()
