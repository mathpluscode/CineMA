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


def run(seed: int) -> None:
    """Run segmentation on LAX 4C images using fine-tuned checkpoint."""
    trained_dataset = "mnms2"

    # load model
    model = ConvUNetR.from_finetuned(
        repo_id="mathpluscode/CineMA",
        model_filename=f"finetuned/segmentation/{trained_dataset}_lax_4c_{seed}.safetensors",
        config_filename="finetuned/segmentation/lax_4c.yaml",
    )

    # load sample data and form a batch of size 1
    transform = ScaleIntensityd(keys="lax_4c")

    # (x, y, t)
    exp_dir = Path(__file__).parent.parent.resolve()
    images = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(exp_dir / "data/ukb/1/1_lax_4c.nii.gz")))
    n_frames = images.shape[-1]
    labels_list = []
    for t in tqdm(range(n_frames), total=n_frames):
        batch = transform({"lax_4c": torch.from_numpy(images[None, ..., t]).to(dtype=torch.float32)})
        batch = {k: v[None, ...] for k, v in batch.items()}  # batch size 1
        with torch.no_grad(), torch.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(batch)["lax_4c"]  # (1, 4, x, y)
        labels = torch.argmax(logits, dim=1)[0].detach().numpy()  # (x, y)

        # the model seems to hallucinate an additional right ventricle and myocardium sometimes
        # find the connected component that is closest to left ventricle
        labels = post_process(labels)

        labels_list.append(labels)
    labels = np.stack(labels_list, axis=-1)  # (x, y, t)

    # visualise segmentations
    _, axs = plt.subplots(10, 5, figsize=(5, 10))
    for i in range(10):
        for j in range(5):
            t = i * 5 + j
            axs[i, j].imshow(images[..., t], cmap="gray")
            axs[i, j].imshow((labels[..., t, None] == 1) * np.array([108 / 255, 142 / 255, 191 / 255, 0.6]))
            axs[i, j].imshow((labels[..., t, None] == 2) * np.array([214 / 255, 182 / 255, 86 / 255, 0.6]))
            axs[i, j].imshow((labels[..., t, None] == 3) * np.array([130 / 255, 179 / 255, 102 / 255, 0.6]))
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            if j == 0:
                axs[i, j].set_ylabel(f"t = {t}")
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig(f"segmentation_lax_4c_mask_{trained_dataset}_{seed}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # visualise area changes
    rv_areas = np.sum(labels == 1, axis=(0, 1))
    myo_areas = np.sum(labels == 2, axis=(0, 1))
    lv_areas = np.sum(labels == 3, axis=(0, 1))
    lvef = (max(lv_areas) - min(lv_areas)) / max(lv_areas) * 100
    rvef = (max(rv_areas) - min(rv_areas)) / max(rv_areas) * 100
    plt.figure(figsize=(4, 3))
    plt.plot(rv_areas, color="#6C8EBF", label="RV")
    plt.plot(myo_areas, color="#D6B656", label="MYO")
    plt.plot(lv_areas, color="#82B366", label="LV")
    plt.xlabel("Frame")
    plt.ylabel("Area (mm2)")
    plt.title(f"LVEF = {lvef:.2f}%, RVEF = {rvef:.2f}%")
    plt.legend(loc="lower right")
    plt.savefig(f"segmentation_lax_4c_mask_area_{trained_dataset}_{seed}.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    for seed in range(3):
        run(seed)
