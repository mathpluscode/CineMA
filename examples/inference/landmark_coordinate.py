"""Example script to perform landmark localization on LAX images using fine-tuned checkpoint."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from monai.transforms import ScaleIntensityd
from tqdm import tqdm

from cinema import ConvViT


def run(view: str, seed: int) -> None:
    """Run landmark localization on LAX images using fine-tuned checkpoint."""
    # load model
    model = ConvViT.from_finetuned(
        repo_id="mathpluscode/CineMA",
        model_filename=f"finetuned/landmark_coordinate/{view}_{seed}.safetensors",
        config_filename=f"finetuned/landmark_coordinate/{view}.yaml",
    )

    # load sample data and form a batch of size 1
    transform = ScaleIntensityd(keys=view)

    # (x, y, t)
    exp_dir = Path(__file__).parent.parent.resolve()
    images = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(exp_dir / f"data/ukb/1/1_{view}.nii.gz")))
    w, h, n_frames = images.shape[-3:]
    preds_list = []
    lv_lengths = []
    for t in tqdm(range(n_frames), total=n_frames):
        batch = transform({view: torch.from_numpy(images[None, ..., t]).to(dtype=torch.float32)})
        batch = {k: v[None, ...] for k, v in batch.items()}  # batch size 1
        with torch.no_grad(), torch.autocast("cuda", enabled=torch.cuda.is_available()):
            coords = model(batch)[0].numpy()  # (6,)
        coords *= np.array([w, h, w, h, w, h])
        coords = [int(x) for x in coords]

        # draw predictions with cross
        preds = images[..., t][..., None] * np.array([1, 1, 1])[None, None, :]
        preds = preds.clip(0, 255).astype(np.uint8)
        for i in range(3):
            pred_x, pred_y = coords[2 * i], coords[2 * i + 1]
            x1, x2 = max(0, pred_x - 9), min(preds.shape[0], pred_x + 10)
            y1, y2 = max(0, pred_y - 9), min(preds.shape[1], pred_y + 10)
            preds[pred_x, y1:y2] = [255, 0, 0]
            preds[x1:x2, pred_y] = [255, 0, 0]
        preds_list.append(preds)

        # record LV length
        x1, y1, x2, y2, x3, y3 = coords
        lv_len = (((x1 + x2) / 2 - x3) ** 2 + ((y1 + y2) / 2 - y3) ** 2) ** 0.5
        lv_lengths.append(lv_len)
    preds = np.stack(preds_list, axis=-1)  # (3, x, y, t)

    # visualise landmarks
    _, axs = plt.subplots(10, 5, figsize=(10, 20))
    for i in range(10):
        for j in range(5):
            t = i * 5 + j
            axs[i, j].imshow(preds[..., t])
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            if j == 0:
                axs[i, j].set_ylabel(f"t = {t}")
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig(f"landmark_coordinate_landmark_{view}_{seed}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # visualise LV length changes
    plt.figure(figsize=(4, 3))
    if view == "lax_2c":
        # first frame is empty for this particular example
        lv_lengths = lv_lengths[1:]
    lvef = (max(lv_lengths) - min(lv_lengths)) / max(lv_lengths) * 100
    plt.plot(lv_lengths, color="#82B366", label="LV")
    plt.xlabel("Frame")
    plt.ylabel("Length (mm)")
    plt.title(f"LVEF = {lvef:.2f}%")
    plt.legend(loc="lower right")
    plt.savefig(f"landmark_coordinate_lv_length_{view}_{seed}.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    for view in ["lax_2c", "lax_4c"]:
        for seed in range(3):
            run(view, seed)
