"""Example script to reconstruct masked patches."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd

from cinema import CineMA, patchify, unpatchify


def run(device: torch.device, dtype: torch.dtype) -> None:
    """Run MAE reconstruction."""
    # load model
    model = CineMA.from_pretrained()
    model.to(device)
    model.eval()

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
    # (x, y, z, t) for SAX and (x, y, 1, t) for LAX
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

    # forward
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
        _, pred_dict, enc_mask_dict, _ = model(batch, enc_mask_ratio=0.75)

    # visualize
    _, axs = plt.subplots(6, 4, figsize=(12, 18))
    for i, view in enumerate(["lax_2c", "lax_3c", "lax_4c", "sax"]):
        patches = patchify(image=batch[view], patch_size=model.dec_patch_size_dict[view])
        patches[enc_mask_dict[view]] = pred_dict[view]
        masks = torch.zeros_like(patches)
        masks[enc_mask_dict[view]] = 1
        masks = unpatchify(
            masks, patch_size=model.dec_patch_size_dict[view], grid_size=model.enc_down_dict[view].patch_embed.grid_size
        )
        masks = masks[0, 0]
        reconstructed = unpatchify(
            patches,
            patch_size=model.dec_patch_size_dict[view],
            grid_size=model.enc_down_dict[view].patch_embed.grid_size,
        )
        reconstructed = reconstructed[0, 0].detach().cpu().numpy()
        image = batch[view][0, 0].detach().cpu().numpy()
        error = np.abs(reconstructed - image)

        if view == "sax":
            for j in range(3):
                z = j * 3
                axs[3 + j, 0].set_ylabel(f"{view} slice {z}")
                axs[3 + j, 0].imshow(image[..., z], cmap="gray")
                axs[3 + j, 1].imshow(masks[..., z], cmap="gray")
                axs[3 + j, 2].imshow(reconstructed[..., z], cmap="gray")
                axs[3 + j, 3].imshow(error[..., z], cmap="gray")
        else:
            axs[i, 0].imshow(image, cmap="gray")
            axs[i, 1].imshow(masks, cmap="gray")
            axs[i, 2].imshow(reconstructed, cmap="gray")
            axs[i, 3].imshow(error, cmap="gray")
            axs[i, 0].set_ylabel(view)
            if i == 0:
                axs[i, 0].set_title("Original")
                axs[i, 1].set_title("Mask")
                axs[i, 2].set_title("Reconstructed")
                axs[i, 3].set_title("Error")
    plt.savefig("mae_reconstruction.png", dpi=300, bbox_inches="tight")
    plt.show(block=False)


if __name__ == "__main__":
    dtype, device = torch.float32, torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

    run(device, dtype)
