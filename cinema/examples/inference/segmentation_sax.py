"""Example script to perform segmentation on SAX images using fine-tuned checkpoint."""

import io
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd
from PIL import Image
from tqdm import tqdm

from cinema import ConvUNetR


def plot_segmentations(images: np.ndarray, labels: np.ndarray, t_step: int, filepath: Path) -> None:
    """Plot segmentations as animated GIF.

    Args:
        images: (x, y, z, t)
        labels: (x, y, z, t)
        t_step: step size for frames
        filepath: path to save the GIF file.
    """
    n_slices, n_frames = labels.shape[-2:]
    n_cols = 3
    n_rows = (n_slices + n_cols - 1) // n_cols  # Calculate rows needed for 3 columns
    frames = []

    for t in tqdm(range(n_frames), desc="Creating segmentation GIF frames"):
        # Create individual frame with SAX slices in grid layout (3 columns)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2), dpi=150)

        # Handle different subplot arrangements
        if n_rows == 1 and n_cols == 1:
            axs = [[axs]]
        elif n_rows == 1:
            axs = [axs]
        elif n_cols == 1:
            axs = [[ax] for ax in axs]

        for z in range(n_slices):
            row = z // n_cols
            col = z % n_cols

            axs[row][col].imshow(images[..., z, t], cmap="gray")
            axs[row][col].imshow((labels[..., z, t, None] == 1) * np.array([108 / 255, 142 / 255, 191 / 255, 0.6]))
            axs[row][col].imshow((labels[..., z, t, None] == 2) * np.array([214 / 255, 182 / 255, 86 / 255, 0.6]))
            axs[row][col].imshow((labels[..., z, t, None] == 3) * np.array([130 / 255, 179 / 255, 102 / 255, 0.6]))
            axs[row][col].set_xticks([])
            axs[row][col].set_yticks([])

        # Hide unused subplots
        for z in range(n_slices, n_rows * n_cols):
            row = z // n_cols
            col = z % n_cols
            axs[row][col].set_visible(False)

        # Reduce spacing between subplots
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.0, hspace=0.0)

        # Render figure to numpy array using BytesIO (universal across backends)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=150)
        buf.seek(0)
        img = Image.open(buf)
        frame = np.array(img.convert("RGB"))
        frames.append(frame)
        buf.close()
        plt.close(fig)

    # Create GIF directly from memory arrays
    with imageio.get_writer(filepath, mode="I", duration=50 * t_step, loop=0) as writer:
        for frame in tqdm(frames, desc="Creating segmentation GIF"):
            writer.append_data(frame)


def plot_volume_changes(labels: np.ndarray, t_step: int, filepath: Path) -> None:
    """Plot volume changes.

    Args:
        labels: (x, y, z, t)
        t_step: step size for frames
        filepath: path to save the PNG file.
    """
    n_frames = labels.shape[-1]
    xs = np.arange(n_frames) * t_step
    rv_volumes = np.sum(labels == 1, axis=(0, 1, 2)) * 10 / 1000
    myo_volumes = np.sum(labels == 2, axis=(0, 1, 2)) * 10 / 1000
    lv_volumes = np.sum(labels == 3, axis=(0, 1, 2)) * 10 / 1000
    lvef = (max(lv_volumes) - min(lv_volumes)) / max(lv_volumes) * 100
    rvef = (max(rv_volumes) - min(rv_volumes)) / max(rv_volumes) * 100

    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ax.plot(xs, rv_volumes, color="#6C8EBF", label="Right Ventricle")
    ax.plot(xs, myo_volumes, color="#D6B656", label="Myocardium")
    ax.plot(xs, lv_volumes, color="#82B366", label="Left Ventricle")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Volume (ml)")
    ax.set_title(f"LVEF = {lvef:.2f}%\nRVEF = {rvef:.2f}%")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    model.eval()
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
    labels = torch.stack(labels_list, dim=-1).detach().to(torch.float32).cpu().numpy()  # (x, y, z, t)

    # visualise segmentations
    plot_segmentations(images, labels, t_step, Path(f"segmentation_{view}_animation_{trained_dataset}_{seed}.gif"))

    # visualise volume changes
    plot_volume_changes(labels, t_step, Path(f"segmentation_{view}_mask_volume_{trained_dataset}_{seed}.png"))


if __name__ == "__main__":
    dtype, device = torch.float32, torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

    for trained_dataset in ["acdc", "mnms", "mnms2"]:
        for seed in range(3):
            run(trained_dataset, seed, device, dtype)
