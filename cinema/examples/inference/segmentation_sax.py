"""Example script to perform segmentation on SAX images using fine-tuned checkpoint."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd
from tqdm import tqdm

from cinema import ConvUNetR


def plot_segmentations(images: np.ndarray, labels: np.ndarray, t_step: int) -> plt.Figure:
    """Plot segmentations.

    Args:
        images: (x, y, z, t)
        labels: (x, y, z, t)
        t_step: step size for frames

    Returns:
        figure
    """
    n_slices, n_frames = labels.shape[-2:]
    fig, axs = plt.subplots(n_frames, n_slices, figsize=(n_slices, n_frames), dpi=300)
    for t in range(n_frames):
        for z in range(n_slices):
            axs[t, z].imshow(images[..., z, t], cmap="gray")
            axs[t, z].imshow((labels[..., z, t, None] == 1) * np.array([108 / 255, 142 / 255, 191 / 255, 0.6]))
            axs[t, z].imshow((labels[..., z, t, None] == 2) * np.array([214 / 255, 182 / 255, 86 / 255, 0.6]))
            axs[t, z].imshow((labels[..., z, t, None] == 3) * np.array([130 / 255, 179 / 255, 102 / 255, 0.6]))
            axs[t, z].set_xticks([])
            axs[t, z].set_yticks([])
            if z == 0:
                axs[t, z].set_ylabel(f"t = {t * t_step}")
    axs[0, n_slices // 2].set_title("SAX Slices")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def plot_volume_changes(labels: np.ndarray, t_step: int) -> plt.Figure:
    """Plot volume changes.

    Args:
        labels: (x, y, z, t)
        t_step: step size for frames

    Returns:
        figure
    """
    n_frames = labels.shape[-1]
    xs = np.arange(n_frames) * t_step
    rv_volumes = np.sum(labels == 1, axis=(0, 1, 2)) * 10 / 1000
    myo_volumes = np.sum(labels == 2, axis=(0, 1, 2)) * 10 / 1000
    lv_volumes = np.sum(labels == 3, axis=(0, 1, 2)) * 10 / 1000
    lvef = (max(lv_volumes) - min(lv_volumes)) / max(lv_volumes) * 100
    rvef = (max(rv_volumes) - min(rv_volumes)) / max(rv_volumes) * 100

    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ax.plot(xs, rv_volumes, color="#6C8EBF", label="RV")
    ax.plot(xs, myo_volumes, color="#D6B656", label="MYO")
    ax.plot(xs, lv_volumes, color="#82B366", label="LV")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Volume (ml)")
    ax.set_title(f"LVEF = {lvef:.2f}%, RVEF = {rvef:.2f}%")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def run(trained_dataset: str, seed: int, device: torch.device, dtype: torch.dtype) -> None:
    """Run segmentation on SAX images using fine-tuned checkpoint."""
    # inference every n frames
    t_step = 5

    # load model
    view = "sax"
    model = ConvUNetR.from_finetuned(
        repo_id="mathpluscode/CineMA",
        model_filename=f"finetuned/segmentation/{trained_dataset}_{view}/{trained_dataset}_{view}_{seed}.safetensors",
        config_filename=f"finetuned/segmentation/{trained_dataset}_{view}/config.yaml",  # same config for all models
    )
    model.to(device)

    # load sample data and form a batch of size 1
    transform = Compose(
        [
            ScaleIntensityd(keys=view),
            SpatialPadd(keys=view, spatial_size=(192, 192, 16), method="end"),
        ]
    )

    # (x, y, z, t)
    exp_dir = Path(__file__).parent.parent.resolve()
    images = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(exp_dir / "data/ukb/1/1_sax.nii.gz")))
    images = images[..., ::t_step]
    n_slices, n_frames = images.shape[-2:]

    labels_list = []
    for t in tqdm(range(n_frames), total=n_frames):
        batch = transform({view: torch.from_numpy(images[None, ..., t])})
        batch = {k: v[None, ...].to(device=device, dtype=torch.float32) for k, v in batch.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
            logits = model(batch)[view]  # (1, 4, x, y, z)
        labels_list.append(torch.argmax(logits, dim=1)[0, ..., :n_slices])
    labels = torch.stack(labels_list, dim=-1).detach().cpu().numpy()  # (x, y, z, t)

    # visualise segmentations
    fig = plot_segmentations(images, labels, t_step)
    fig.savefig(f"segmentation_{view}_mask_{trained_dataset}_{seed}.png", dpi=300, bbox_inches="tight")
    plt.show(block=False)

    # visualise volume changes
    fig = plot_volume_changes(labels, t_step)
    fig.savefig(f"segmentation_{view}_mask_volume_{trained_dataset}_{seed}.png", dpi=300, bbox_inches="tight")
    plt.show(block=False)


if __name__ == "__main__":
    dtype, device = torch.float32, torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

    for trained_dataset in ["acdc", "mnms", "mnms2"]:
        for seed in range(3):
            run(trained_dataset, seed, device, dtype)
