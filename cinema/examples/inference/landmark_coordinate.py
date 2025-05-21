"""Example script to perform landmark localization on LAX images using fine-tuned checkpoint."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from monai.transforms import ScaleIntensityd
from tqdm import tqdm

from cinema import ConvViT
from cinema.examples.inference.landmark_heatmap import plot_landmarks, plot_lv


def run(view: str, seed: int, device: torch.device, dtype: torch.dtype) -> None:
    """Run landmark localization on LAX images using fine-tuned checkpoint."""
    # load model
    model = ConvViT.from_finetuned(
        repo_id="mathpluscode/CineMA",
        model_filename=f"finetuned/landmark_coordinate/{view}/{view}_{seed}.safetensors",
        config_filename=f"finetuned/landmark_coordinate/{view}/config.yaml",
    )
    model.eval()
    model.to(device)

    # load sample data and form a batch of size 1
    transform = ScaleIntensityd(keys=view)

    # (x, y, 1, t)
    exp_dir = Path(__file__).parent.parent.resolve()
    images = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(exp_dir / f"data/ukb/1/1_{view}.nii.gz")))
    w, h, _, n_frames = images.shape
    coords_list = []
    for t in tqdm(range(n_frames), total=n_frames):
        batch = transform({view: torch.from_numpy(images[None, ..., 0, t])})
        batch = {k: v[None, ...].to(device=device, dtype=dtype) for k, v in batch.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
            coords = model(batch)[0].numpy()  # (6,)
        coords *= np.array([w, h, w, h, w, h])
        coords = [int(x) for x in coords]
        coords_list.append(coords)
    coords = np.stack(coords_list, axis=-1)  # (6, t)

    # visualise landmarks
    fig = plot_landmarks(images, coords)
    fig.savefig(f"landmark_coordinate_landmark_{view}_{seed}.png", dpi=300, bbox_inches="tight")
    plt.show(block=False)

    # visualise LV length changes
    fig = plot_lv(coords)
    plt.savefig(f"landmark_coordinate_gls_{view}_{seed}.png", dpi=300, bbox_inches="tight")
    plt.show(block=False)


if __name__ == "__main__":
    dtype, device = torch.float32, torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

    for view in ["lax_2c", "lax_4c"]:
        for seed in range(3):
            run(view, seed, device, dtype)
