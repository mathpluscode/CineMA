"""Evaluation functions for classification task."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader, SequentialSampler

from cinema.classification.dataset import EndDiastoleEndSystoleDataset, get_image_transforms
from cinema.classification.train import (
    classification_eval,
    classification_metrics,
    get_classification_or_regression_model,
)
from cinema.device import get_amp_dtype_and_device
from cinema.log import get_logger

if TYPE_CHECKING:
    from omegaconf import DictConfig


logger = get_logger(__name__)


def classification_eval_dataset(  # pylint:disable=too-many-statements
    config: DictConfig,
    split: str,
    ckpt_path: Path,
    out_dir: Path,
) -> None:
    """Function to evaluate classification model on end-diastole and end-systole dataset.

    Args:
        config: config for evaluation.
        split: split of data, train or test.
        ckpt_path: path to the checkpoint.
        out_dir: output directory.
    """
    # load data
    data_dir = Path(config.data.dir)
    if not (data_dir / f"{split}_metadata.csv").exists():
        logger.warning(f"{split}_metadata.csv does not exist. Skip evaluation.")
        return
    meta_df = pd.read_csv(data_dir / f"{split}_metadata.csv", dtype={"pid": str})
    # certain class may not exist
    class_col = config.data.class_column
    classes = config.data[class_col]
    n = len(meta_df)
    meta_df = meta_df[meta_df[class_col].isin(classes)].reset_index(drop=True)
    if len(meta_df) < n:
        logger.warning(f"Removed {n - len(meta_df)} samples from {split} split.")
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
        class_col=class_col,
        classes=classes,
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
    pred_logits = []
    pids = []
    for _, batch in enumerate(dataloader):
        logits, _ = classification_eval(
            model=model, batch=batch, patch_size_dict=patch_size_dict, amp_dtype=amp_dtype, device=device
        )
        pred_labels.append(torch.argmax(logits, dim=1))
        true_labels.append(batch["label"])
        pred_logits.append(logits)
        pids += batch["pid"]
    pred_labels = torch.cat(pred_labels, dim=0).cpu().to(dtype=torch.float32).numpy()
    true_labels = torch.cat(true_labels, dim=0).cpu().to(dtype=torch.float32).numpy()
    pred_logits = torch.cat(pred_logits, dim=0).cpu().to(dtype=torch.float32)
    pred_probs = F.softmax(pred_logits, dim=1).numpy()  # softmax after dtype conversion to ensure sum=1
    metrics = classification_metrics(
        true_labels=true_labels,
        pred_labels=pred_labels,
        pred_probs=pred_probs,
    )
    pred_df = pd.DataFrame(
        {
            "pid": pids,
            "true_label": true_labels.tolist(),
            "pred_label": pred_labels.tolist(),
            "pred_probability": pred_probs.tolist(),
        }
    )
    (out_dir / split).mkdir(exist_ok=True, parents=True)
    pred_df.sort_values("pid").to_csv(out_dir / split / "classification_prediction.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / split / "classification_metrics.csv", index=False)
    logger.info(f"Classification metrics: {metrics}")
