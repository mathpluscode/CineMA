"""Script to parse raw files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import rescale
from tqdm import tqdm

from cinema.log import get_logger

logger = get_logger(__name__)

VAL_PROPORTION = 0.2
TEST_PROPORTION = 0.2


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Folder of data.",
        default=None,
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Folder for output.",
        default=None,
    )
    args = parser.parse_args()

    return args


def get_orientation(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> str:
    """Get orientation of three points.

    Args:
        x1: x-coordinate of point 1.
        y1: y-coordinate of point 1.
        x2: x-coordinate of point 2.
        y2: y-coordinate of point 2.
        x3: x-coordinate of point 3.
        y3: y-coordinate of point 3.

    Returns:
        orientation of the points.
    """
    val = (y2 - y1) * (x3 - x2) - (x2 - x1) * (y3 - y2)
    if val > 0:
        return "clockwise"
    if val < 0:
        return "counterclockwise"
    return "collinear"


def get_landmark_coordinates(mask: np.ndarray) -> dict[str, int]:
    """Get landmark coordinates from mask.

    mask only have zeros 0, 85, 170, 255, corresponding to background and landmark 1, 2, 3

    Args:
        mask: mask of the image.

    Returns:
        dict: landmark coordinates.
    """
    coords = {}
    for i, v in zip([1, 2, 3], [85, 170, 255], strict=False):
        xs, ys = np.nonzero(mask == v)
        x, y = int(np.mean(xs)), int(np.mean(ys))
        coords[f"x{i}"] = x
        coords[f"y{i}"] = y
    return coords


def draw_landmarks(image: np.ndarray, coords: dict[str, int], out_path: Path) -> None:
    """Draw landmarks on the image.

    Args:
        image: (width, height, 3) image.
        coords: metrics having ground truth landmarks.
        out_path: output path.
    """
    for i in range(1, 4):
        # draw ground truth with red cross
        x, y = coords[f"x{i}"], coords[f"y{i}"]
        x1, x2 = max(0, x - 5), min(image.shape[0], x + 6)
        y1, y2 = max(0, y - 5), min(image.shape[1], y + 6)
        image[x, y1:y2] = [255, 0, 0]
        image[x1:x2, y] = [255, 0, 0]
    out_path.parent.mkdir(exist_ok=True, parents=True)
    Image.fromarray(np.moveaxis(image, 0, 1)).save(out_path)


def process(data_dir: Path, out_dir: Path, scale: float = 0.25) -> None:
    """Derive landmark coordinates from masks.

    The actions in this function are:
    1. Load mask to get landmark coordinates.
    2. Determine if needs to flip the image on y-axis.
    3. If needs to flip, flip the masks, and recalculate landmark coordinates.
    4. Rescale each image and flip if needed.
    2. Store the derived landmark coordinates and rescaled images.

    Args:
        data_dir: folder of data.
        out_dir: folder for output.
        scale: scale factor for rescaling.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for view in ["lax_2c", "lax_4c"]:
        meta_df = pd.read_csv(
            data_dir / f"{view}.csv", header=None, names=["cohort_name", "uid", "view", "landmark_number", "x", "y"]
        )
        uid_df = meta_df[["cohort_name", "uid", "view"]].drop_duplicates()
        data = []
        for _, row in tqdm(uid_df.iterrows(), total=len(uid_df)):
            uid = row["uid"]

            # after transpose, mask.shape = (width, height) = (x, y)
            mask = np.transpose(np.array(Image.open(data_dir / view / "masks" / f"{uid}.png")))
            coords = get_landmark_coordinates(mask)

            # lax 2c is mainly clockwise, and lax 4c is mainly counterclockwise
            orientation = get_orientation(
                coords["x1"], coords["y1"], coords["x2"], coords["y2"], coords["x3"], coords["y3"]
            )
            flip = (view == "lax_2c" and orientation != "clockwise") or (
                view == "lax_4c" and orientation != "counterclockwise"
            )
            if flip:
                mask = np.flip(mask, axis=1)
                coords = get_landmark_coordinates(mask)

            # load image, resize, and flip
            image = np.transpose(np.array(Image.open(data_dir / view / "images" / f"{uid}.png")))
            image = rescale(image, scale=scale, anti_aliasing=True)
            if flip:
                image = np.flip(image, axis=1)
            img_dir = out_dir / view / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            image = image[..., None] * np.array([255, 255, 255])[None, None, :]
            image = image.clip(0, 255).astype(np.uint8)
            Image.fromarray(np.moveaxis(image, 0, 1)).convert("L").save(img_dir / f"{uid}.png")

            # rescale coordinates
            coords = {k: int(v * scale) for k, v in coords.items()}

            # draw coordinates for visualization
            img_dir = out_dir / view / "images_with_landmarks"
            img_dir.mkdir(parents=True, exist_ok=True)
            draw_landmarks(image, coords, img_dir / f"{uid}.png")

            # store metadata
            data.append(
                {
                    "cohort_name": row["cohort_name"],
                    "uid": uid,
                    "view": row["view"],
                    **coords,
                }
            )
        meta_df = pd.DataFrame(data)
        meta_df.to_csv(out_dir / f"{view}.csv", index=False)

        # split data
        meta_df["pid"] = meta_df["uid"].apply(lambda x: x.split("_")[0])
        pids = meta_df["pid"].unique()
        test_size = int(len(pids) * TEST_PROPORTION)
        val_size = int(len(pids) * VAL_PROPORTION)
        train_size = len(pids) - test_size - val_size
        logger.info(f"Using {train_size} samples for training, {val_size} for validation, and {test_size} for testing.")
        rng = np.random.default_rng(0)
        rng.shuffle(pids)
        train_pids = pids[:train_size]
        val_pids = pids[train_size : train_size + val_size]
        test_pids = pids[train_size + val_size :]
        train_meta_df = meta_df[meta_df["pid"].isin(train_pids)]
        val_meta_df = meta_df[meta_df["pid"].isin(val_pids)]
        test_meta_df = meta_df[meta_df["pid"].isin(test_pids)]
        train_meta_df.to_csv(out_dir / f"{view}_train.csv", index=False)
        val_meta_df.to_csv(out_dir / f"{view}_val.csv", index=False)
        test_meta_df.to_csv(out_dir / f"{view}_test.csv", index=False)


def main() -> None:
    """Main function."""
    args = parse_args()
    process(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
