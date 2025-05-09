"""Example script to perform BMI regression using pre-trained checkpoint."""

from pathlib import Path

import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from huggingface_hub import hf_hub_download
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd
from omegaconf import OmegaConf

from cinema import ConvViT


def run(seed: int) -> None:
    """Run BMI regression using fine-tuned checkpoint."""
    trained_dataset, view = "acdc", "sax"
    # load config to get class names
    config_path = hf_hub_download(
        repo_id="mathpluscode/CineMA",
        filename=f"finetuned/regression_bmi/{trained_dataset}_{view}.yaml",
    )
    config = OmegaConf.load(config_path)
    mean = config.data[config.data.regression_column].mean
    std = config.data[config.data.regression_column].std

    # load model
    model = ConvViT.from_finetuned(
        repo_id="mathpluscode/CineMA",
        model_filename=f"finetuned/regression_bmi/{trained_dataset}_{view}_{seed}.safetensors",
        config_filename=f"finetuned/regression_bmi/{trained_dataset}_{view}.yaml",
    )

    # load sample data
    spatial_size = (192, 192, 16) if view == "sax" else (256, 256)
    transform = Compose(
        [
            ScaleIntensityd(keys=view),
            SpatialPadd(keys=view, spatial_size=spatial_size, method="end", lazy=True, allow_missing_keys=True),
        ]
    )
    exp_dir = Path(__file__).parent.parent.resolve()
    ed_image = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(exp_dir / f"data/acdc/{view}_ed.nii.gz")))
    es_image = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(exp_dir / f"data/acdc/{view}_es.nii.gz")))
    image = np.stack([ed_image, es_image], axis=0)  # (2, x, y, 1) or (2, x, y, z)
    if view != "sax":
        image = image[..., 0]  # (2, x, y, 1) -> (2, x, y)
    batch = transform({view: torch.from_numpy(image).to(dtype=torch.float32)})
    batch = {k: v[None, ...] for k, v in batch.items()}  # batch size 1
    with torch.no_grad(), torch.autocast("cuda", enabled=torch.cuda.is_available()):
        preds = model(batch) * std + mean
    print(f"Using {view} view with model trained on {trained_dataset} dataset with seed {seed}.")  # noqa: T201
    print(f"The predicted BMI is {preds[0][0]}, ground truth should be 27.66.")  # noqa: T201


if __name__ == "__main__":
    for seed in range(3):
        run(seed)
