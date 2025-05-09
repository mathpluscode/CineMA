"""Example script to perform segmentation on SAX images using fine-tuned checkpoint."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd
from tqdm import tqdm

from cinema import ConvUNetR


def run(trained_dataset: str, seed: int) -> None:
    """Run segmentation on SAX images using fine-tuned checkpoint."""
    # load model
    model = ConvUNetR.from_finetuned(
        repo_id="mathpluscode/CineMA",
        model_filename=f"finetuned/segmentation/{trained_dataset}_sax_{seed}.safetensors",
        config_filename="finetuned/segmentation/sax.yaml",  # same config for all models
    )

    # load sample data and form a batch of size 1
    transform = Compose(
        [
            ScaleIntensityd(keys="sax"),
            SpatialPadd(keys="sax", spatial_size=(192, 192, 16), method="end", lazy=True, allow_missing_keys=True),
        ]
    )

    # (x, y, z, t)
    exp_dir = Path(__file__).parent.parent.resolve()
    images = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(exp_dir / "data/ukb/1/1_sax.nii.gz")))
    n_slices, n_frames = images.shape[-2:]
    labels_list = []
    for t in tqdm(range(n_frames), total=n_frames):
        batch = transform({"sax": torch.from_numpy(images[None, ..., t]).to(dtype=torch.float32)})
        batch = {k: v[None, ...] for k, v in batch.items()}  # batch size 1
        with torch.no_grad(), torch.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(batch)["sax"]  # (1, 4, x, y, z)
        labels_list.append(torch.argmax(logits, dim=1)[0, ..., :n_slices])
    labels = torch.stack(labels_list, dim=-1).detach().numpy()  # (x, y, z, t)

    # visualise segmentations
    _, axs = plt.subplots(9, 9, figsize=(9, 9))
    for i in range(9):
        t = i * 5
        for z in range(9):
            axs[i, z].imshow(images[..., z, t], cmap="gray")
            axs[i, z].imshow((labels[..., z, t, None] == 1) * np.array([108 / 255, 142 / 255, 191 / 255, 0.6]))
            axs[i, z].imshow((labels[..., z, t, None] == 2) * np.array([214 / 255, 182 / 255, 86 / 255, 0.6]))
            axs[i, z].imshow((labels[..., z, t, None] == 3) * np.array([130 / 255, 179 / 255, 102 / 255, 0.6]))
            axs[i, z].set_xticks([])
            axs[i, z].set_yticks([])
            if z == 0:
                axs[i, z].set_ylabel(f"t = {t}")
    axs[0, 4].set_title("SAX Slices")
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig(f"segmentation_sax_mask_{trained_dataset}_{seed}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # visualise volume changes
    rv_volumes = np.sum(labels == 1, axis=(0, 1, 2)) * 10 / 1000  # ml = 1000 mm^3
    myo_volumes = np.sum(labels == 2, axis=(0, 1, 2)) * 10 / 1000
    lv_volumes = np.sum(labels == 3, axis=(0, 1, 2)) * 10 / 1000
    lvef = (max(lv_volumes) - min(lv_volumes)) / max(lv_volumes) * 100
    rvef = (max(rv_volumes) - min(rv_volumes)) / max(rv_volumes) * 100
    plt.figure(figsize=(4, 3))
    plt.plot(rv_volumes, color="#6C8EBF", label="RV")
    plt.plot(myo_volumes, color="#D6B656", label="MYO")
    plt.plot(lv_volumes, color="#82B366", label="LV")
    plt.xlabel("Frame")
    plt.ylabel("Volume (ml)")
    plt.title(f"LVEF = {lvef:.2f}%, RVEF = {rvef:.2f}%")
    plt.legend(loc="lower right")
    plt.savefig(f"segmentation_sax_mask_volume_{trained_dataset}_{seed}.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    for trained_dataset in ["acdc", "mnms", "mnms2"]:
        for seed in range(3):
            run(trained_dataset, seed)
