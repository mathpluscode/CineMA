"""Evaluation functions for classification task."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler

from cinema.classification.dataset import get_image_transforms
from cinema.classification.train import get_classification_or_regression_model
from cinema.device import get_amp_dtype_and_device
from cinema.log import get_logger
from cinema.regression.dataset import EndDiastoleEndSystoleDataset
from cinema.regression.train import regression_eval, regression_metrics

if TYPE_CHECKING:
    from omegaconf import DictConfig


logger = get_logger(__name__)


def regression_eval_dataset(  # pylint:disable=too-many-statements
    config: DictConfig,
    split: str,
    ckpt_path: Path,
    out_dir: Path,
) -> None:
    """Function to evaluate regression model on end-diastole and end-systole dataset.

    Args:
        config: config for evaluation.
        split: split of data, train or test.
        ckpt_path: path to the checkpoint.
        out_dir: output directory.
        dataset_cls: dataset class for evaluation.
    """
    out_dir = out_dir / split
    out_dir.mkdir(exist_ok=True, parents=True)
    # load data
    data_dir = Path(config.data.dir)
    if not (data_dir / f"{split}_metadata.csv").exists():
        logger.warning(f"{split}_metadata.csv does not exist. Skip evaluation.")
        return
    meta_df = pd.read_csv(data_dir / f"{split}_metadata.csv", dtype={"pid": str})
    reg_col = config.data.regression_column
    reg_mean = config.data[reg_col].mean
    reg_std = config.data[reg_col].std
    if config.data.max_n_samples > 0:
        meta_df = meta_df.sample(n=min(config.data.max_n_samples, len(meta_df)), random_state=0)
        logger.info(f"Using {config.data.max_n_samples} samples for {split} split.")
    views = [config.model.views] if isinstance(config.model.views, str) else config.model.views
    patch_size_dict = {v: config.data.sax.patch_size if v == "sax" else config.data.lax.patch_size for v in views}
    _, transform = get_image_transforms(config)

    dataset = EndDiastoleEndSystoleDataset(
        data_dir=data_dir / split,
        meta_df=meta_df,
        views=views,
        reg_col=reg_col,
        reg_mean=reg_mean,
        reg_std=reg_std,
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
    model = get_classification_or_regression_model(config)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # inference
    pred_labels = []
    true_labels = []
    pids = []
    for _, batch in enumerate(dataloader):
        preds, _ = regression_eval(
            model=model, batch=batch, patch_size_dict=patch_size_dict, amp_dtype=amp_dtype, device=device
        )
        pred_labels.append(preds)
        true_labels.append(batch["label"])
        pids += batch["pid"]
    pred_labels = torch.cat(pred_labels, dim=0).cpu().to(dtype=torch.float32).numpy()
    true_labels = torch.cat(true_labels, dim=0).cpu().to(dtype=torch.float32).numpy()
    restored_pred_labels = (pred_labels * reg_std) + reg_mean
    restored_true_labels = (true_labels * reg_std) + reg_mean
    metrics = regression_metrics(
        true_labels=true_labels,
        pred_labels=pred_labels,
    )
    restored_metrics = regression_metrics(
        true_labels=restored_true_labels,
        pred_labels=restored_pred_labels,
    )
    restored_metrics = {f"restored_{k}": v for k, v in restored_metrics.items()}
    metrics.update(restored_metrics)
    pd.DataFrame([metrics]).to_csv(out_dir / "regression_metrics.csv", index=False)
    logger.info(f"Regression metrics: {metrics}")

    pred_df = pd.DataFrame(
        {
            "pid": pids,
            "true_label": true_labels.tolist(),
            "pred_label": pred_labels.tolist(),
            "restored_true_label": restored_true_labels.tolist(),
            "restored_pred_label": restored_pred_labels.tolist(),
        }
    )
    pred_df.sort_values("pid").to_csv(out_dir / "regression_prediction.csv", index=False)
    logger.info(f"Saved regression predictions to {out_dir / 'regression_prediction.csv'}")
