"""Example script to perform segmentation on LAX 4C images using fine-tuned checkpoint."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from monai.transforms import ScaleIntensityd
from scipy.spatial.distance import cdist
from skimage import measure
from tqdm import tqdm

from cinema import ConvUNetR


def post_process(labels: np.ndarray) -> np.ndarray:
    """Remove artifact by choosing predictions closet to RV."""
    processed = np.zeros_like(labels)

    # rv no processing
    rv_mask = labels == 1
    processed[rv_mask] = 1

    # myo and lv
    for i in [2, 3]:
        mask = labels == i
        labeled_mask = measure.label(mask)
        bc = np.bincount(labeled_mask.flat, weights=mask.flat)

        # find closest component
        closest = None
        d_max = np.inf
        for j, c in enumerate(bc):
            if c > 0:
                d = cdist(np.argwhere(rv_mask), np.argwhere(labeled_mask == j), "minkowski", p=1.0).min()
                if d < d_max:
                    closest = j
        processed[labeled_mask == closest] = i

    return processed


def plot_segmentations(images: np.ndarray, labels: np.ndarray, n_cols: int = 5) -> plt.Figure:
    """Plot segmentations.

    Args:
        images: (x, y, t)
        labels: (x, y, t)
        n_cols: number of columns

    Returns:
        figure
    """
    n_frames = labels.shape[-1]
    n_rows = n_frames // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows), dpi=300)
    for i in range(n_rows):
        for j in range(n_cols):
            t = i * n_cols + j
            axs[i, j].imshow(images[..., 0, t], cmap="gray")
            axs[i, j].imshow((labels[..., t, None] == 1) * np.array([108 / 255, 142 / 255, 191 / 255, 0.6]))
            axs[i, j].imshow((labels[..., t, None] == 2) * np.array([214 / 255, 182 / 255, 86 / 255, 0.6]))
            axs[i, j].imshow((labels[..., t, None] == 3) * np.array([130 / 255, 179 / 255, 102 / 255, 0.6]))
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            if j == 0:
                axs[i, j].set_ylabel(f"t = {t}")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def plot_volume_changes(labels: np.ndarray) -> plt.Figure:
    """Plot volume changes.

    Args:
        labels: (x, y, t)

    Returns:
        figure
    """
    n_frames = labels.shape[-1]
    xs = np.arange(n_frames)
    rv_volumes = np.sum(labels == 1, axis=(0, 1)) * 10 / 1000
    myo_volumes = np.sum(labels == 2, axis=(0, 1)) * 10 / 1000
    lv_volumes = np.sum(labels == 3, axis=(0, 1)) * 10 / 1000
    lvef = (max(lv_volumes) - min(lv_volumes)) / max(lv_volumes) * 100
    rvef = (max(rv_volumes) - min(rv_volumes)) / max(rv_volumes) * 100

    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ax.plot(xs, rv_volumes, color="#6C8EBF", label="RV")
    ax.plot(xs, myo_volumes, color="#D6B656", label="MYO")
    ax.plot(xs, lv_volumes, color="#82B366", label="LV")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Area (mm2)")
    ax.set_title(f"LVEF = {lvef:.2f}%, RVEF = {rvef:.2f}%")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def run(trained_dataset: str, seed: int, device: torch.device, dtype: torch.dtype) -> None:
    """Run segmentation on LAX 4C images using fine-tuned checkpoint."""
    # load model
    view = "lax_4c"
    model = ConvUNetR.from_finetuned(
        repo_id="mathpluscode/CineMA",
        model_filename=f"finetuned/segmentation/{trained_dataset}_{view}/{trained_dataset}_{view}_{seed}.safetensors",
        config_filename=f"finetuned/segmentation/{trained_dataset}_{view}/config.yaml",
    )
    model.eval()
    model.to(device)

    # load sample data and form a batch of size 1
    transform = ScaleIntensityd(keys=view)

    # (x, y, 1, t)
    exp_dir = Path(__file__).parent.parent.resolve()
    images = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(exp_dir / "data/ukb/1/1_lax_4c.nii.gz")))

    n_frames = images.shape[-1]
    labels_list = []
    for t in tqdm(range(n_frames), total=n_frames):
        batch = transform({view: torch.from_numpy(images[None, ..., 0, t])})
        batch = {k: v[None, ...].to(device=device, dtype=dtype) for k, v in batch.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
            logits = model(batch)[view]  # (1, 4, x, y)
        labels = torch.argmax(logits, dim=1)[0].detach().cpu().numpy()  # (x, y)

        # the model seems to hallucinate an additional right ventricle and myocardium sometimes
        # find the connected component that is closest to left ventricle
        labels = post_process(labels)
        labels_list.append(labels)
    labels = np.stack(labels_list, axis=-1)  # (x, y, t)

    # visualise segmentations
    fig = plot_segmentations(images, labels)
    plt.savefig(f"segmentation_{view}_mask_{trained_dataset}_{seed}.png", dpi=300, bbox_inches="tight")
    plt.show(block=False)

    # visualise area changes
    fig = plot_volume_changes(labels)
    fig.savefig(f"segmentation_{view}_mask_area_{trained_dataset}_{seed}.png", dpi=300, bbox_inches="tight")
    plt.show(block=False)


if __name__ == "__main__":
    dtype, device = torch.float32, torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

    trained_dataset = "mnms2"
    for seed in range(3):
        run(trained_dataset, seed, device, dtype)
