"""Example script to perform segmentation on LAX 4C images using fine-tuned checkpoint."""

from pathlib import Path

import imageio
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


def plot_segmentations(images: np.ndarray, labels: np.ndarray, filepath: Path) -> None:
    """Plot segmentations as animated GIF.

    Args:
        images: (x, y, 1, t)
        labels: (x, y, t)
        filepath: path to save the GIF file.
    """
    n_frames = labels.shape[-1]
    temp_frame_paths = []

    for t in tqdm(range(n_frames), desc="Creating GIF frames"):
        # Create individual frame
        fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

        # Plot image
        ax.imshow(images[..., 0, t], cmap="gray")

        # Plot segmentation overlays
        ax.imshow((labels[..., t, None] == 1) * np.array([108 / 255, 142 / 255, 191 / 255, 0.6]))
        ax.imshow((labels[..., t, None] == 2) * np.array([214 / 255, 182 / 255, 86 / 255, 0.6]))
        ax.imshow((labels[..., t, None] == 3) * np.array([130 / 255, 179 / 255, 102 / 255, 0.6]))

        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])

        # Save frame
        frame_path = f"_tmp_frame_{t:03d}.png"
        plt.savefig(frame_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig)
        temp_frame_paths.append(frame_path)

    # Create GIF
    with imageio.get_writer(filepath, mode="I", duration=100, loop=0) as writer:
        for frame_path in tqdm(temp_frame_paths, desc="Creating GIF"):
            image = imageio.v2.imread(frame_path)
            writer.append_data(image)
            # Clean up temporary file
            Path(frame_path).unlink()


def plot_volume_changes(labels: np.ndarray, filepath: Path) -> None:
    """Plot volume changes.

    Args:
        labels: (x, y, t)
        filepath: path to save the PNG file.
    """
    n_frames = labels.shape[-1]
    xs = np.arange(n_frames)
    rv_volumes = np.sum(labels == 1, axis=(0, 1)) * 10 / 1000
    myo_volumes = np.sum(labels == 2, axis=(0, 1)) * 10 / 1000
    lv_volumes = np.sum(labels == 3, axis=(0, 1)) * 10 / 1000
    lvef = (max(lv_volumes) - min(lv_volumes)) / max(lv_volumes) * 100
    rvef = (max(rv_volumes) - min(rv_volumes)) / max(rv_volumes) * 100

    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ax.plot(xs, rv_volumes, color="#6C8EBF", label="Right Ventricle")
    ax.plot(xs, myo_volumes, color="#D6B656", label="Myocardium")
    ax.plot(xs, lv_volumes, color="#82B366", label="Left Ventricle")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Area (mm2)")
    ax.set_title(f"LVEF = {lvef:.2f}%\nRVEF = {rvef:.2f}%")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
        labels = torch.argmax(logits, dim=1)[0].detach().to(torch.float32).cpu().numpy()  # (x, y)

        # the model seems to hallucinate an additional right ventricle and myocardium sometimes
        # find the connected component that is closest to left ventricle
        labels = post_process(labels)
        labels_list.append(labels)
    labels = np.stack(labels_list, axis=-1)  # (x, y, t)

    # visualise segmentations
    plot_segmentations(images, labels, Path(f"segmentation_{view}_animation_{trained_dataset}_{seed}.gif"))

    # visualise area changes
    plot_volume_changes(labels, Path(f"segmentation_{view}_mask_area_{trained_dataset}_{seed}.png"))


if __name__ == "__main__":
    dtype, device = torch.float32, torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

    trained_dataset = "mnms2"
    for seed in range(3):
        run(trained_dataset, seed, device, dtype)
