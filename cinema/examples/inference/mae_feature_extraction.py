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
            ScaleIntensityd(keys="sax"),
            SpatialPadd(keys="sax", spatial_size=(192, 192, 16), method="end"),
        ]
    )
    exp_dir = Path(__file__).parent.parent.resolve()
    sax_image = torch.from_numpy(
        np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(exp_dir / "data/acdc/sax_t.nii.gz")))
    )
    t = 0  # which time frame to use
    batch = {
        "sax": sax_image[None, ..., t],
    }
    batch = transform(batch)
    print(f"SAX view had originally {sax_image.shape[-2]} slices, now zero-padded to {batch['sax'].shape[-1]} slices.")  # noqa: T201
    batch = {k: v[None, ...].to(device=device, dtype=dtype) for k, v in batch.items()}

    # forward with sax view
    print("Feature extraction with SAX view")  # noqa: T201
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
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
