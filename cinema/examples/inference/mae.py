"""Example script to reconstruct masked patches."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd

from cinema import CineMA, patchify, unpatchify


def plot_mae_reconstruction(
    batch: dict[str, torch.Tensor],
    pred_dict: dict[str, torch.Tensor],
    enc_mask_dict: dict[str, torch.Tensor],
    patch_size_dict: dict[str, tuple[int, ...]],
    grid_size_dict: dict[str, tuple[int, ...]],
    sax_slices: int,
) -> plt.Figure:
    """Plot MAE reconstruction."""
    n_rows = sax_slices + 3
    n_cols = 4
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2), dpi=300)
    for i, view in enumerate(["lax_2c", "lax_3c", "lax_4c", "sax"]):
        patches = patchify(image=batch[view], patch_size=patch_size_dict[view])
        patches[enc_mask_dict[view]] = pred_dict[view]
        masks = torch.zeros_like(patches)
        masks[enc_mask_dict[view]] = 1
        masks = unpatchify(masks, patch_size=patch_size_dict[view], grid_size=grid_size_dict[view])
        masks = masks[0, 0]
        reconstructed = unpatchify(
            patches,
            patch_size=patch_size_dict[view],
            grid_size=grid_size_dict[view],
        )
        reconstructed = reconstructed[0, 0].numpy()
        image = batch[view][0, 0].numpy()
        error = np.abs(reconstructed - image)

        if view == "sax":
            reconstructed = reconstructed[..., :sax_slices]
            for j in range(sax_slices):
                axs[3 + j, 0].set_ylabel(f"SAX slice {j}")
                axs[3 + j, 0].imshow(image[..., j], cmap="gray")
                axs[3 + j, 1].imshow(masks[..., j], cmap="gray")
                axs[3 + j, 2].imshow(reconstructed[..., j], cmap="gray")
                axs[3 + j, 3].imshow(error[..., j], cmap="gray")
        else:
            axs[i, 0].imshow(image, cmap="gray")
            axs[i, 1].imshow(masks, cmap="gray")
            axs[i, 2].imshow(reconstructed, cmap="gray")
            axs[i, 3].imshow(error, cmap="gray")
            axs[i, 0].set_ylabel({"lax_2c": "LAX 2C", "lax_3c": "LAX 3C", "lax_4c": "LAX 4C"}[view])
            if i == 0:
                axs[i, 0].set_title("Original")
                axs[i, 1].set_title("Mask")
                axs[i, 2].set_title("Reconstructed")
                axs[i, 3].set_title("Error")
    # remove the x and y ticks
    for i in range(n_rows):
        for j in range(n_cols):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def run(device: torch.device, dtype: torch.dtype) -> None:
    """Run MAE reconstruction."""
    t = 25  # which time frame to use

    # load model
    model = CineMA.from_pretrained()
    model.eval()
    patch_size_dict = model.dec_patch_size_dict
    grid_size_dict = {k: v.patch_embed.grid_size for k, v in model.enc_down_dict.items()}
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
    sax_slices = sax_image.shape[-2]
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
    batch = {k: v.detach().cpu() for k, v in batch.items()}
    fig = plot_mae_reconstruction(batch, pred_dict, enc_mask_dict, patch_size_dict, grid_size_dict, sax_slices)
    fig.savefig("mae_reconstruction.png", dpi=300, bbox_inches="tight")
    plt.show(block=False)


if __name__ == "__main__":
    dtype, device = torch.float32, torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

    run(device, dtype)
