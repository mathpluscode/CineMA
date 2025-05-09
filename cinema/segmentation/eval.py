"""Evaluation functions for segmentation task."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef, root_mean_squared_error
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from cinema import LV_LABEL
from cinema.data.sitk import save_image
from cinema.device import get_amp_dtype_and_device
from cinema.log import get_logger
from cinema.metric import ejection_fraction, get_ef_region
from cinema.segmentation.dataset import EndDiastoleEndSystoleDataset, get_segmentation_transforms
from cinema.segmentation.train import get_segmentation_model, segmentation_eval, segmentation_metrics

if TYPE_CHECKING:
    from omegaconf import DictConfig
logger = get_logger(__name__)


def get_ejection_fraction(metric_df: pd.DataFrame, views: list[str]) -> pd.DataFrame:
    """Get ejection fraction from metric dataframe.

    Args:
        metric_df: dataframe, each row is one ED or ES slice.
        views: list of views.

    Returns:
        dataframe, each row is one patient.
    """
    columns = ["pid", f"class_{LV_LABEL}_true_volume", f"class_{LV_LABEL}_pred_volume"]
    for view in views:
        columns += [f"{view}_class_{LV_LABEL}_true_volume", f"{view}_class_{LV_LABEL}_pred_volume"]
    ed_metric_df = metric_df[metric_df["is_ed"]][columns].set_index("pid").add_prefix("ed_")
    es_metric_df = metric_df[~metric_df["is_ed"]][columns].set_index("pid").add_prefix("es_")
    ef_df = ed_metric_df.merge(es_metric_df, on="pid")

    prefixes = [f"{view}_" for view in views]
    for p in ["", *prefixes]:
        ef_df = ef_df.rename(
            columns={
                f"ed_{p}class_{LV_LABEL}_true_volume": f"{p}true_edv",
                f"ed_{p}class_{LV_LABEL}_pred_volume": f"{p}pred_edv",
                f"es_{p}class_{LV_LABEL}_true_volume": f"{p}true_esv",
                f"es_{p}class_{LV_LABEL}_pred_volume": f"{p}pred_esv",
            },
            errors="raise",
        )
        ef_df[f"{p}true_ef"] = ejection_fraction(
            edv=ef_df[f"{p}true_edv"].to_numpy(),
            esv=ef_df[f"{p}true_esv"].to_numpy(),
        )
        ef_df[f"{p}pred_ef"] = ejection_fraction(
            edv=ef_df[f"{p}pred_edv"].to_numpy(),
            esv=ef_df[f"{p}pred_esv"].to_numpy(),
        )
        ef_df[f"{p}pred_ef"] = ef_df[f"{p}pred_ef"].fillna(0).replace([float("inf"), float("-inf")], 0).clip(0, 100)
        ef_df[f"{p}true_ef"] = ef_df[f"{p}true_ef"].fillna(0).replace([float("inf"), float("-inf")], 0).clip(0, 100)
        ef_df[f"{p}ef_error"] = abs(ef_df[f"{p}pred_ef"] - ef_df[f"{p}true_ef"])
    return ef_df


def process_mean_metrics(
    metric_df: pd.DataFrame,
    metric_path: Path,
) -> None:
    """Process mean metrics and save to csv.

    Args:
        metric_df: dataframe, each row is one ED or ES slice.
        metric_path: path to save mean metrics.
    """
    mean_metrics = {f"{k}_mean": v for k, v in metric_df.drop(columns=["pid", "is_ed"]).mean().to_dict().items()}
    std_metrics = {f"{k}_std": v for k, v in metric_df.drop(columns=["pid", "is_ed"]).std().to_dict().items()}
    pd.DataFrame([{**mean_metrics, **std_metrics}]).T.to_csv(metric_path, header=False)
    logger.info(f"Saved mean metrics to {metric_path}.")


def process_ef_metrics(
    ef_df: pd.DataFrame,
    views: list[str],
    metric_path: Path,
) -> None:
    """Process ejection fraction metrics and save to csv.

    Args:
        ef_df: dataframe, each row is one patient, containing true_ef, pred_ef, true_edv, pred_edv, true_esv, pred_esv.
        views: list of views.
        metric_path: path to save ejection fraction metrics.
    """
    prefixes = [f"{view}_" for view in views]
    metrics = {}
    for prefix in ["", *prefixes]:
        ef_true_labels = ef_df[f"{prefix}true_ef"].apply(get_ef_region)
        ef_pred_labels = ef_df[f"{prefix}pred_ef"].apply(get_ef_region)
        ef_metrics = {
            f"{prefix}ef_mae": ef_df[f"{prefix}ef_error"].mean(),
            f"{prefix}edv_mae": (ef_df[f"{prefix}true_edv"] - ef_df[f"{prefix}pred_edv"]).abs().mean(),
            f"{prefix}esv_mae": (ef_df[f"{prefix}true_esv"] - ef_df[f"{prefix}pred_esv"]).abs().mean(),
            f"{prefix}ef_err_std": ef_df[f"{prefix}ef_error"].std(),
            f"{prefix}edv_err_std": (ef_df[f"{prefix}true_edv"] - ef_df[f"{prefix}pred_edv"]).abs().std(),
            f"{prefix}esv_err_std": (ef_df[f"{prefix}true_esv"] - ef_df[f"{prefix}pred_esv"]).abs().std(),
            f"{prefix}ef_rmse": root_mean_squared_error(ef_df[f"{prefix}true_ef"], ef_df[f"{prefix}pred_ef"]),
            f"{prefix}edv_rmse": root_mean_squared_error(ef_df[f"{prefix}true_edv"], ef_df[f"{prefix}pred_edv"]),
            f"{prefix}esv_rmse": root_mean_squared_error(ef_df[f"{prefix}true_esv"], ef_df[f"{prefix}pred_esv"]),
            f"{prefix}ef_acc": accuracy_score(y_true=ef_true_labels, y_pred=ef_pred_labels),
            f"{prefix}ef_mcc": matthews_corrcoef(y_true=ef_true_labels, y_pred=ef_pred_labels),
        }
        metrics.update(ef_metrics)
    pd.DataFrame([metrics]).T.to_csv(metric_path, header=False)


def save_segmentation_metrics(metric_df: pd.DataFrame, views: list[str], out_dir: Path) -> None:
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

    logger.info("Mean metrics across ED and ES slices.")
    process_mean_metrics(metric_df, out_dir / "mean_metrics.csv")

    # ejection fraction
    metric_path = out_dir / "ef_metrics.csv"
    ef_df = get_ejection_fraction(metric_df, views=views)
    ef_df.to_csv(metric_path, index=True)
    logger.info(f"Saved ejection fraction metrics to {metric_path}.")

    logger.info("Ejection fraction metrics across patients.")
    process_ef_metrics(ef_df, views=views, metric_path=out_dir / "mean_ef_metrics.csv")


def segmentation_eval_edes_dataset(
    config: DictConfig,
    split: str,
    ckpt_path: Path,
    out_dir: Path,
    save: bool,
) -> None:
    """Function to evaluate segmentation model on end-diastole and end-systole dataset.

    Args:
        config: config for evaluation.
        split: split of data, train or test.
        ckpt_path: path to the checkpoint.
        out_dir: output directory.
        save: whether to save the predictions.
    """
    # load data
    data_dir = Path(config.data.dir)
    meta_df = pd.read_csv(data_dir / f"{split}_metadata.csv", dtype={"pid": str})
    if config.data.max_n_samples > 0:
        meta_df = meta_df.sample(n=min(config.data.max_n_samples, len(meta_df)), random_state=0)
        logger.info(f"Using {config.data.max_n_samples} samples for {split} split.")
    views = [config.model.views] if isinstance(config.model.views, str) else config.model.views
    patch_size_dict = {
        view: config.data.sax.patch_size if view == "sax" else config.data.lax.patch_size for view in views
    }
    spacing_dict = {view: config.data.sax.spacing if view == "sax" else config.data.lax.spacing for view in views}
    _, transform = get_segmentation_transforms(config)
    dataset = EndDiastoleEndSystoleDataset(
        data_dir=data_dir / split,
        meta_df=meta_df,
        views=views,
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
    checkpoint = torch.load(ckpt_path, map_location="cpu")
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
            metrics_fn=segmentation_metrics,
        )

        # save metrics
        pid = batch["pid"][0]
        is_ed = bool(batch["is_ed"].numpy())
        frame_name = "ed" if is_ed else "es"
        metrics["pid"] = pid
        metrics["is_ed"] = is_ed
        metrics_list.append(metrics)

        # save segmentation
        if not save:
            continue
        # (n_classes, *image_size)
        pred_probs_dict = {
            view: torch.softmax(logits_dict[view], dim=1)[0].cpu().to(dtype=torch.float32).numpy() for view in views
        }
        # (*image_size)
        pred_labels_dict = {view: np.argmax(pred_probs_dict[view], axis=0).astype(np.uint8) for view in views}
        for view in views:
            true_label_path = data_dir / split / pid / f"{pid}_{view}_{frame_name}_gt.nii.gz"
            pred_label_path = out_dir / split / pid / f"{pid}_{view}_{frame_name}_pred.nii.gz"
            image_np = pred_labels_dict[view] if view == "sax" else pred_labels_dict[view][..., None]
            save_image(
                image_np=image_np,
                reference_image_path=true_label_path,
                out_path=pred_label_path,
            )
        pred_probs_path = out_dir / split / pid / f"{pid}_{frame_name}_probs.npz"
        np.savez_compressed(pred_probs_path, **pred_probs_dict)

    # save metrics per sample
    metric_df = pd.DataFrame(metrics_list).sort_values("pid")
    save_segmentation_metrics(metric_df, views=views, out_dir=out_dir / split)
