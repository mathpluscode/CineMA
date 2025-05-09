"""Evaluation of segmentation model for Kaggle dataset."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, matthews_corrcoef, root_mean_squared_error
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from cinema import LV_LABEL
from cinema.data.kaggle import KAGGLE_SPACING
from cinema.data.sitk import save_image
from cinema.device import get_amp_dtype_and_device
from cinema.log import get_logger
from cinema.metric import ejection_fraction, get_ef_region
from cinema.segmentation.eval import process_ef_metrics
from cinema.segmentation.kaggle.dataset import KaggleVideoDataset
from cinema.segmentation.train import get_segmentation_model, segmentation_forward

logger = get_logger(__name__)

MAX_N_FRAMES = 30


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
        help="Directory of Kaggle data.",
        required=True,
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        help="Split of data, train or validate or test.",
        default="test",
    )
    parser.add_argument("--save", action="store_true", help="Save the predictions.")
    parser.set_defaults(save=False)
    args = parser.parse_args()

    return args


def save_metrics(metric_df: pd.DataFrame, out_dir: Path) -> None:
    """Save metrics to csv.

    Args:
        metric_df: dataframe, each row is one ED or ES slice.
        out_dir: output directory.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    metric_path = out_dir / "metrics.csv"
    metric_df.to_csv(metric_path, index=False)
    logger.info(f"Saved metrics to {metric_path}.")

    # save mean metrics
    metric_path = out_dir / "mean_metrics.csv"
    mean_metrics = {f"{k}_mean": v for k, v in metric_df.drop(columns=["pid"]).mean().to_dict().items()}
    std_metrics = {f"{k}_std": v for k, v in metric_df.drop(columns=["pid"]).std().to_dict().items()}

    # 0, reduced; 1, borderline; 2, normal
    ef_true_labels = metric_df["true_ef"].apply(get_ef_region)
    ef_pred_labels = metric_df["pred_ef"].apply(get_ef_region)
    ef_metrics = {
        "ef_mae": (metric_df["true_ef"] - metric_df["pred_ef"]).abs().mean(),
        "edv_mae": (metric_df["true_edv"] - metric_df["pred_edv"]).abs().mean(),
        "esv_mae": (metric_df["true_esv"] - metric_df["pred_esv"]).abs().mean(),
        "ef_err_std": (metric_df["true_ef"] - metric_df["pred_ef"]).abs().std(),
        "edv_err_std": (metric_df["true_edv"] - metric_df["pred_edv"]).abs().std(),
        "esv_err_std": (metric_df["true_esv"] - metric_df["pred_esv"]).abs().std(),
        "ef_rmse": root_mean_squared_error(metric_df["true_ef"], metric_df["pred_ef"]),
        "edv_rmse": root_mean_squared_error(metric_df["true_edv"], metric_df["pred_edv"]),
        "esv_rmse": root_mean_squared_error(metric_df["true_esv"], metric_df["pred_esv"]),
        "ef_acc": accuracy_score(y_true=ef_true_labels, y_pred=ef_pred_labels),
        "ef_mcc": matthews_corrcoef(y_true=ef_true_labels, y_pred=ef_pred_labels),
    }
    pd.DataFrame([{**mean_metrics, **std_metrics, **ef_metrics}]).T.to_csv(metric_path, header=False)
    logger.info(f"Saved mean metrics to {metric_path}.")

    # print markdown table
    ef_mae = ef_metrics["ef_mae"]
    ef_rmse = ef_metrics["ef_rmse"]
    edv_rmse = ef_metrics["edv_rmse"]
    esv_rmse = ef_metrics["esv_rmse"]
    columns = "| EF MAE (%) | EF RMSE (%) | EDV RMSE | ESV RMSE |"
    values = f"| {ef_mae:.2f} | {ef_rmse:.2f} | {edv_rmse:.2f} | {esv_rmse:.2f} |"
    logger.info(columns)
    logger.info(values)


def segmentation_eval_kaggle_dataset(  # pylint:disable=too-many-statements
    config: DictConfig,
    split: str,
    ckpt_path: Path,
    out_dir: Path,
    save: bool,
) -> None:
    """Evaluate the model.

    Args:
        config: config for training.
        split: train or validate or test.
        ckpt_path: path to the checkpoint file.
        out_dir: output directory.
        save: whether to save the predictions.
    """
    data_dir = config.data.dir
    view = config.model.views
    if not isinstance(view, str):
        raise TypeError("Only support one view for evaluation.")
    # load Kaggle data
    meta_df = pd.read_csv(data_dir / f"{split}_metadata.csv")
    if config.data.max_n_samples > 0:
        meta_df = meta_df.sample(n=min(config.data.max_n_samples, len(meta_df)), random_state=0)
        logger.info(f"Using {config.data.max_n_samples} samples for validation.")
    patch_size_dict = {view: config.data.sax.patch_size if view == "sax" else config.data.lax.patch_size}
    transform = Compose(
        [
            ScaleIntensityd(keys=f"{view}_image", allow_missing_keys=True),
            SpatialPadd(
                keys=f"{view}_image",
                spatial_size=patch_size_dict[view],
                method="end",
                lazy=True,
            ),
        ]
    )
    dataset = KaggleVideoDataset(
        data_dir=data_dir / split, meta_df=meta_df, view=view, max_n_frames=MAX_N_FRAMES, transform=transform
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

    # load segmentation model
    amp_dtype, device = get_amp_dtype_and_device()
    model = get_segmentation_model(config)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # inference
    lv_metrics_list = []
    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        pid = batch["pid"].numpy()[0]
        image = batch[f"{view}_image"].to(device)  # (1, n_frames, 1, x, y, z) for SAX or (1, n_frames, 1, x, y) for LAX
        n_frames = min(batch["n_frames"].numpy()[0], MAX_N_FRAMES)
        n_slices = batch["n_slices"].numpy()[0]

        lv_volumes = []
        preds_labels = []
        for t in range(n_frames):
            logits_dict = segmentation_forward(
                model=model, image_dict={view: image[:, t]}, patch_size_dict=patch_size_dict, amp_dtype=amp_dtype
            )
            pred_labels = torch.argmax(logits_dict[view], dim=1)[0].cpu()  # (x, y, z) for SAX or (x, y) for LAX
            preds_labels.append(pred_labels)
            lv_volume = (pred_labels == LV_LABEL).sum().item() * math.prod(KAGGLE_SPACING) / 1000.0  # ml = 1000 mm^3
            lv_volumes.append(lv_volume)

        # calculate metrics use LV only
        pred_edv, pred_esv = max(lv_volumes), min(lv_volumes)
        pred_ef = ejection_fraction(pred_edv, pred_esv) if pred_edv > 0 else 0.0
        lv_metrics_list.append(
            {
                "pid": pid,
                "true_edv": batch["edv"].numpy()[0],
                "true_esv": batch["esv"].numpy()[0],
                "true_ef": batch["ef"].numpy()[0],
                "pred_edv": pred_edv,
                "pred_esv": pred_esv,
                "pred_ef": pred_ef,
                "pred_ed_index": np.argmax(lv_volumes),
                "pred_es_index": np.argmin(lv_volumes),
            }
        )

        # save segmentation
        if not save:
            continue

        # (x, y, z, n_frames) for SAX or (x, y, n_frames) for LAX
        pred_labels = torch.stack(preds_labels, dim=-1).numpy().astype(np.uint8)
        if view == "sax":
            n_pad_slices = pred_labels.shape[-2] - n_slices
            start_idx = n_pad_slices // 2
            end_idx = n_slices + start_idx
            pred_labels = pred_labels[:, :, start_idx:end_idx, :]
        true_label_path = data_dir / split / str(pid) / f"{pid}_{view}_t.nii.gz"
        pred_label_path = out_dir / split / str(pid) / f"{pid}_{view}_t_pred.nii.gz"
        save_image(
            image_np=pred_labels,
            reference_image_path=true_label_path,
            out_path=pred_label_path,
        )

    # save metrics per sample
    metric_df = pd.DataFrame(lv_metrics_list).sort_values("pid")
    metric_df["pred_ef"] = metric_df["pred_ef"].fillna(0).replace([float("inf"), float("-inf")], 0).clip(0, 100)
    metric_df["ef_error"] = (metric_df["true_ef"] - metric_df["pred_ef"]).abs()
    save_metrics(metric_df, out_dir / split)

    logger.info("Ejection fraction metrics across patients.")
    process_ef_metrics(metric_df, views=[], metric_path=out_dir / split / "mean_ef_metrics.csv")


def main() -> None:
    """Main function."""
    args = parse_args()

    out_dir = args.ckpt_path.parent / f"kaggle_eval_{args.ckpt_path.stem}"
    out_dir.mkdir(exist_ok=True)

    config = OmegaConf.load(args.ckpt_path.parent / "config.yaml")
    config.data.dir = args.data_dir
    segmentation_eval_kaggle_dataset(
        config=config,
        split=args.split,
        ckpt_path=args.ckpt_path,
        out_dir=out_dir,
        save=args.save,
    )


if __name__ == "__main__":
    main()
