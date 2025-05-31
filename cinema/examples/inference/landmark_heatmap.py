"""Example script to perform landmark localization on LAX images using fine-tuned checkpoint."""

import io
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from monai.transforms import ScaleIntensityd
from PIL import Image
from tqdm import tqdm

from cinema import ConvUNetR, heatmap_soft_argmax


def plot_heatmaps(images: np.ndarray, probs: np.ndarray, filepath: Path) -> None:
    """Plot heatmaps as animated GIF.

    Args:
        images: (x, y, 1, t)
        probs: (3, x, y, t)
        filepath: path to save the GIF file.
    """
    n_frames = probs.shape[-1]
    frames = []

    for t in tqdm(range(n_frames), desc="Creating heatmap GIF frames"):
        # Create individual frame
        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

        # Plot image
        ax.imshow(images[..., 0, t], cmap="gray")

        # Plot heatmap overlays
        ax.imshow(probs[0, ..., t, None] * np.array([1.0, 0.0, 0.0, 1.0]))
        ax.imshow(probs[1, ..., t, None] * np.array([1.0, 0.0, 0.0, 1.0]))
        ax.imshow(probs[2, ..., t, None] * np.array([1.0, 0.0, 0.0, 1.0]))

        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])

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
    with imageio.get_writer(filepath, mode="I", duration=50, loop=0) as writer:
        for frame in tqdm(frames, desc="Creating heatmap GIF"):
            writer.append_data(frame)


def plot_landmarks(images: np.ndarray, coords: np.ndarray, filepath: Path) -> None:
    """Plot landmarks as animated GIF.

    Args:
        images: (x, y, 1, t)
        coords: (6, t)
        filepath: path to save the GIF file.
    """
    n_frames = images.shape[-1]
    frames = []

    for t in tqdm(range(n_frames), desc="Creating landmark GIF frames"):
        # Create individual frame
        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

        # draw predictions with cross
        preds = images[..., t] * np.array([1, 1, 1])[None, None, :]
        preds = preds.clip(0, 255).astype(np.uint8)
        for k in range(3):
            pred_x, pred_y = coords[2 * k, t], coords[2 * k + 1, t]
            x1, x2 = max(0, pred_x - 9), min(preds.shape[0], pred_x + 10)
            y1, y2 = max(0, pred_y - 9), min(preds.shape[1], pred_y + 10)
            preds[pred_x, y1:y2] = [255, 0, 0]
            preds[x1:x2, pred_y] = [255, 0, 0]

        ax.imshow(preds)
        ax.set_xticks([])
        ax.set_yticks([])

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
    with imageio.get_writer(filepath, mode="I", duration=50, loop=0) as writer:
        for frame in tqdm(frames, desc="Creating landmark GIF"):
            writer.append_data(frame)


def plot_lv(coords: np.ndarray, filepath: Path) -> None:
    """Plot GL shortening.

    Args:
        coords: (6, t)
        filepath: path to save the PNG file.
    """
    # GL shortening
    x1, y1 = coords[0], coords[1]
    x2, y2 = coords[2], coords[3]
    x3, y3 = coords[4], coords[5]
    lv_lengths = (((x1 + x2) / 2 - x3) ** 2 + ((y1 + y2) / 2 - y3) ** 2) ** 0.5
    gls = (max(lv_lengths) - min(lv_lengths)) / max(lv_lengths) * 100

    # MAPSE
    ed_idx = np.argmin(lv_lengths)
    es_idx = np.argmax(lv_lengths)
    x1_ed, y1_ed = coords[0, ed_idx], coords[1, ed_idx]
    x2_ed, y2_ed = coords[2, ed_idx], coords[3, ed_idx]
    x1_es, y1_es = coords[0, es_idx], coords[1, es_idx]
    x2_es, y2_es = coords[2, es_idx], coords[3, es_idx]
    mapse = (
        ((x1_ed - x1_es) ** 2 + (y1_ed - y1_es) ** 2) ** 0.5 + ((x2_ed - x2_es) ** 2 + (y2_ed - y2_es) ** 2) ** 0.5
    ) / 2

    fig = plt.figure(figsize=(4, 4), dpi=120)
    plt.plot(lv_lengths, color="#82B366", label="Left Ventricle")
    plt.xlabel("Frame")
    plt.ylabel("Length (mm)")
    plt.title(f"GLS = {gls:.2f}%\nMAPSE = {mapse:.2f} mm")
    plt.legend(loc="lower right")
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_and_landmarks(images: np.ndarray, probs: np.ndarray, coords: np.ndarray, filepath: Path) -> None:
    """Plot combined heatmap and landmarks as animated GIF.

    Args:
        images: (x, y, 1, t)
        probs: (3, x, y, t)
        coords: (6, t)
        filepath: path to save the GIF file.
    """
    n_frames = probs.shape[-1]
    frames = []

    for t in tqdm(range(n_frames), desc="Creating combined GIF frames"):
        # Create single frame with image + colored heatmaps + landmarks
        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

        # Plot original image as background
        ax.imshow(images[..., 0, t], cmap="gray")

        # Plot heatmap
        ax.imshow(probs[0, ..., t, None] * np.array([1, 0, 0, 0.6]))
        ax.imshow(probs[1, ..., t, None] * np.array([1, 0, 0, 0.6]))
        ax.imshow(probs[2, ..., t, None] * np.array([1, 0, 0, 0.6]))

        # Add landmark crosses on top
        for k in range(3):
            pred_x, pred_y = coords[2 * k, t], coords[2 * k + 1, t]
            ax.plot([pred_y - 9, pred_y + 9], [pred_x, pred_x], color="red", linewidth=2)
            ax.plot([pred_y, pred_y], [pred_x - 9, pred_x + 9], color="red", linewidth=2)

        ax.set_xticks([])
        ax.set_yticks([])

        # Render figure to numpy array using BytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=150)
        buf.seek(0)
        img = Image.open(buf)
        frame = np.array(img.convert("RGB"))
        frames.append(frame)
        buf.close()
        plt.close(fig)

    # Create GIF directly from memory arrays
    with imageio.get_writer(filepath, mode="I", duration=100, loop=0) as writer:
        for frame in tqdm(frames, desc="Creating combined GIF"):
            writer.append_data(frame)


def run(view: str, seed: int, device: torch.device, dtype: torch.dtype) -> None:
    """Run landmark localization on LAX images using fine-tuned checkpoint."""
    # load model
    model = ConvUNetR.from_finetuned(
        repo_id="mathpluscode/CineMA",
        model_filename=f"finetuned/landmark_heatmap/{view}/{view}_{seed}.safetensors",
        config_filename=f"finetuned/landmark_heatmap/{view}/config.yaml",
    )
    model.eval()
    model.to(device)

    # load sample data and form a batch of size 1
    transform = ScaleIntensityd(keys=view)

    # (x, y, 1, t)
    exp_dir = Path(__file__).parent.parent.resolve()
    images = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(exp_dir / f"data/ukb/1/1_{view}.nii.gz")))
    n_frames = images.shape[-1]
    probs_list = []
    coords_list = []
    for t in tqdm(range(n_frames), total=n_frames):
        batch = transform({view: torch.from_numpy(images[None, ..., 0, t])})
        batch = {k: v[None, ...].to(device=device, dtype=dtype) for k, v in batch.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
            logits = model(batch)[view]  # (1, 3, x, y)
        probs = torch.sigmoid(logits)  # (1, 3, width, height)
        probs_list.append(probs[0].detach().to(torch.float32).cpu().numpy())
        coords = heatmap_soft_argmax(probs)[0].numpy()
        coords = [int(x) for x in coords]
        coords_list.append(coords)
    probs = np.stack(probs_list, axis=-1)  # (3, x, y, t)
    coords = np.stack(coords_list, axis=-1)  # (6, t)

    # visualise heatmaps
    plot_heatmaps(images, probs, Path(f"landmark_heatmap_probs_{view}_{seed}.gif"))

    # visualise landmarks
    plot_landmarks(images, coords, Path(f"landmark_heatmap_landmark_{view}_{seed}.gif"))

    # visualise LV length changes
    plot_lv(coords, Path(f"landmark_heatmap_gls_{view}_{seed}.png"))

    # visualise heatmap and landmarks
    plot_heatmap_and_landmarks(images, probs, coords, Path(f"landmark_heatmap_probs_and_landmark_{view}_{seed}.gif"))


if __name__ == "__main__":
    dtype, device = torch.float32, torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

    for view in ["lax_2c", "lax_4c"]:
        for seed_idx in range(3):
            run(view, seed_idx, device, dtype)
