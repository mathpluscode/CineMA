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

from cinema import LABEL_TO_NAME, LV_LABEL
from cinema.data.sitk import save_image
from cinema.device import get_amp_dtype_and_device
from cinema.log import get_logger
from cinema.metric import coefficient_of_variance, ejection_fraction
from cinema.segmentation.dataset import get_segmentation_transforms
from cinema.segmentation.eval import process_ef_metrics
from cinema.segmentation.rescan.dataset import CineSegmentationDataset
from cinema.segmentation.train import get_segmentation_model, segmentation_eval, segmentation_metrics

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
        choices=["train", "val", "test"],
        help="Split of data, train or test.",
        default="test",
    )
    parser.add_argument("--save", action="store_true", help="Save the predictions.")
    parser.set_defaults(save=False)
    args = parser.parse_args()

    return args


def process_mean_metrics(
    metric_df: pd.DataFrame,
    metric_path: Path,
) -> None:
    """Process mean metrics and save to csv.

    Args:
        metric_df: dataframe, each row is one ED or ES slice.
        metric_path: path to save mean metrics.
    """
    mean_metrics = {f"{k}_mean": v for k, v in metric_df.drop(columns=["pid", "frame"]).mean().to_dict().items()}
    std_metrics = {f"{k}_std": v for k, v in metric_df.drop(columns=["pid", "frame"]).std().to_dict().items()}
    pd.DataFrame([{**mean_metrics, **std_metrics}]).T.to_csv(metric_path, header=False)
    logger.info(f"Saved mean metrics to {metric_path}.")
    # print markdown table
    for metric_name, metric_scale in zip(["dice_score", "hausdorff_distance_95"], [100, 1], strict=False):
        metric_names = [f"mean_{metric_name}"]
        metric_names += [f"class_{i}_{metric_name}" for i in range(1, len(LABEL_TO_NAME) + 1)]
        column_names = [f"mean_{metric_name}"]
        column_names += [f"{LABEL_TO_NAME[i].lower()}_{metric_name}" for i in range(1, len(LABEL_TO_NAME) + 1)]
        columns = "| " + " | ".join(column_names) + " |"
        columns = columns.replace("hausdorff_distance_95", "hd")
        values = "|"
        for x in metric_names:
            values += (
                f" {mean_metrics[f'{x}_mean'] * metric_scale:.2f} ({std_metrics[f'{x}_std'] * metric_scale:.2f}) |"
            )
        logger.info(columns)
        logger.info(values)


def get_ejection_fraction(metric_df: pd.DataFrame) -> pd.DataFrame:
    """Get ejection fraction from metric dataframe.

    This function is necessary as each slice is not just ED or ES slice, and
    the metric_df does not have is_ed column.

    Args:
        metric_df: dataframe, each row is one ED or ES slice or other slices.

    Returns:
        dataframe, each row is one patient.
    """
    columns = [f"class_{LV_LABEL}_true_volume", f"class_{LV_LABEL}_pred_volume"]
    ed_metric_df = metric_df.groupby("pid")[columns].agg("max").add_prefix("ed_")
    es_metric_df = metric_df.groupby("pid")[columns].agg("min").add_prefix("es_")
    metric_df = ed_metric_df.merge(es_metric_df, on="pid")
    metric_df = metric_df.rename(
        columns={
            f"ed_class_{LV_LABEL}_true_volume": "true_edv",
            f"ed_class_{LV_LABEL}_pred_volume": "pred_edv",
            f"es_class_{LV_LABEL}_true_volume": "true_esv",
            f"es_class_{LV_LABEL}_pred_volume": "pred_esv",
        },
        errors="raise",
    )
    metric_df["true_ef"] = ejection_fraction(
        edv=metric_df["true_edv"].to_numpy(),
        esv=metric_df["true_esv"].to_numpy(),
    )
    metric_df["pred_ef"] = ejection_fraction(
        edv=metric_df["pred_edv"].to_numpy(),
        esv=metric_df["pred_esv"].to_numpy(),
    )
    metric_df["pred_ef"] = metric_df["pred_ef"].fillna(0).replace([float("inf"), float("-inf")], 0).clip(0, 100)
    metric_df["true_ef"] = metric_df["true_ef"].fillna(0).replace([float("inf"), float("-inf")], 0).clip(0, 100)
    metric_df["ef_error"] = abs(metric_df["pred_ef"] - metric_df["true_ef"])
    return metric_df


def get_coefficient_of_variance(ef_df: pd.DataFrame, metric_path: Path) -> None:
    """Get coefficient of variance.

    Args:
        ef_df: dataframe with EF metrics, index is pid.
        metric_path: path to save coefficient of variance.

    Returns:
        dict with coefficient of variance.
    """
    ef_df = ef_df.reset_index(drop=False)
    ef_df["repeat"] = ef_df["pid"].apply(lambda x: x.split("_")[2])
    ef_df["pid"] = ef_df["pid"].apply(lambda x: int(x.split("_")[1]))

    pred_df = ef_df.pivot_table(index="pid", columns="repeat", values="pred_ef").dropna()
    true_df = ef_df.pivot_table(index="pid", columns="repeat", values="true_ef").dropna()

    cv_pred = coefficient_of_variance(pred_df["A"].to_numpy(), pred_df["B"].to_numpy())
    cv_true = coefficient_of_variance(true_df["A"].to_numpy(), true_df["B"].to_numpy())
    logger.info(f"Coefficient of variance for EF: {cv_pred * 100:.2f} (pred), {cv_true * 100:.2f} (true).")

    pd.DataFrame(
        {
            "cv_pred": [cv_pred],
            "cv_true": [cv_true],
        }
    ).to_csv(metric_path, index=False)


def save_segmentation_metrics(metric_df: pd.DataFrame, out_dir: Path) -> None:
    """Save metrics to csv.

    Args:
        metric_df: dataframe, each row is one ED or ES slice.
        views: list of views.
        out_dir: output directory.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    metric_path = out_dir / "metrics.csv"
    metric_df.to_csv(metric_path, index=False)
    logger.info(f"Saved metrics to {metric_path}.")

    process_mean_metrics(metric_df, out_dir / "mean_metrics.csv")

    metric_path = out_dir / "ef_metrics.csv"
    ef_df = get_ejection_fraction(metric_df)
    ef_df.to_csv(metric_path, index=True)
    logger.info(f"Saved ejection fraction metrics to {metric_path}.")

    logger.info("Ejection fraction metrics across patients.")
    process_ef_metrics(ef_df, views=[], metric_path=out_dir / "mean_ef_metrics.csv")
    get_coefficient_of_variance(ef_df, out_dir / "ef_cv.csv")


def segmentation_eval_rescan_dataset(
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
        has_labels=True,
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
        logits_dict, metrics = segmentation_eval(
            model=model,
            batch=batch,
            patch_size_dict=patch_size_dict,
            spacing_dict=spacing_dict,
            amp_dtype=amp_dtype,
            device=device,
            metrics_fn=segmentation_metrics,
        )

        # save metrics
        pid = batch["pid"][0]
        frame = int(batch["frame"].numpy())  # frame of the slice
        metrics["pid"] = pid
        metrics["n_slices"] = int(batch["n_slices"].numpy())
        metrics["frame"] = frame
        metrics_list.append(metrics)

        # save segmentation
        if not save:
            continue
        pred_labels_dict = {
            view: torch.argmax(logits_dict[view][0], dim=0).cpu().to(dtype=torch.uint8).numpy() for view in views
        }
        for view in views:
            true_label_path = data_dir / split / pid / f"{view}_gt_t.nii.gz"
            pred_label_path = out_dir / split / pid / f"{view}_pred_t{frame}.nii.gz"
            save_image(
                image_np=pred_labels_dict[view],
                reference_image_path=true_label_path,
                out_path=pred_label_path,
            )

    # save metrics per sample
    metric_df = pd.DataFrame(metrics_list).sort_values("pid")
    save_segmentation_metrics(metric_df, out_dir=out_dir / split)


def main() -> None:
    """Main function."""
    args = parse_args()

    out_dir = args.ckpt_path.parent / f"rescan_eval_{args.ckpt_path.stem}"
    out_dir.mkdir(exist_ok=True)

    # load and overwrite config
    config = OmegaConf.load(args.ckpt_path.parent / "config.yaml")
    if args.data_dir is not None:
        config.data.dir = Path(args.data_dir).expanduser()

    segmentation_eval_rescan_dataset(
        config=config,
        split=args.split,
        ckpt_path=args.ckpt_path,
        out_dir=out_dir,
        save=args.save,
    )


if __name__ == "__main__":
    main()
