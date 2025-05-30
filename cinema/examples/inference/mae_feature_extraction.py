"""Example script to extract features from CineMA."""

from pathlib import Path

import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd

from cinema import CineMA


def run(device: torch.device, dtype: torch.dtype) -> None:
    """Run MAE feature extraction."""
    # load model
    model = CineMA.from_pretrained()
    model.eval()
    model.to(device)

    # load sample data and form a batch of size 1
    transform = Compose(
        [
            ScaleIntensityd(keys=("sax", "lax_2c", "lax_3c", "lax_4c"), allow_missing_keys=True),
            SpatialPadd(keys="sax", spatial_size=(192, 192, 16), method="end"),
            SpatialPadd(
                keys=("lax_2c", "lax_3c", "lax_4c"),
                spatial_size=(256, 256),
                method="end",
                lazy=True,
                allow_missing_keys=True,
            ),
        ]
    )
    exp_dir = Path(__file__).parent.parent.resolve()
    sax_image = torch.from_numpy(
        np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(exp_dir / "data/ukb/1/1_sax.nii.gz")))
    )
    lax_2c_image = torch.from_numpy(
        np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(exp_dir / "data/ukb/1/1_lax_2c.nii.gz")))
    )
    lax_3c_image = torch.from_numpy(
        np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(exp_dir / "data/ukb/1/1_lax_3c.nii.gz")))
    )
    lax_4c_image = torch.from_numpy(
        np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(exp_dir / "data/ukb/1/1_lax_4c.nii.gz")))
    )
    t = 25  # which time frame to use
    batch = {
        "sax": sax_image[None, ..., t],
        "lax_2c": lax_2c_image[None, ..., 0, t],
        "lax_3c": lax_3c_image[None, ..., 0, t],
        "lax_4c": lax_4c_image[None, ..., 0, t],
    }
    batch = transform(batch)
    print(f"SAX view had originally {sax_image.shape[-2]} slices, now zero-padded to {batch['sax'].shape[-1]} slices.")  # noqa: T201
    batch = {k: v[None, ...].to(device=device, dtype=dtype) for k, v in batch.items()}

    # forward with all views
    print("Feature extraction with all views")  # noqa: T201
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
        feature_dict = model.feature_forward(batch)
    for k, v in feature_dict.items():
        print(k, v.shape)  # noqa: T201

    # forward with two views
    print("Feature extraction with two views")  # noqa: T201
    batch = {"sax": batch["sax"], "lax_4c": batch["lax_4c"]}
    feature_dict = model.feature_forward(batch)
    for k, v in feature_dict.items():
        print(k, v.shape)  # noqa: T201


if __name__ == "__main__":
    dtype, device = torch.float32, torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

    run(device, dtype)
