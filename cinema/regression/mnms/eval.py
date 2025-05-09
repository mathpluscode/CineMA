"""Script to evaluate."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from cinema.log import get_logger
from cinema.regression.eval import regression_eval_dataset

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to the checkpoint .pt file, config is assumed to be saved in the same directory.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Data directory, if not provided, using the one saved in config.",
        default=None,
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        help="Split of data, train or test.",
        default="test",
    )
    args = parser.parse_args()

    return args


def main() -> None:
    """Main function."""
    args = parse_args()

    out_dir = args.ckpt_path.parent / f"mnms_eval_{args.ckpt_path.stem}"
    out_dir.mkdir(exist_ok=True)

    # load and overwrite config
    config = OmegaConf.load(args.ckpt_path.parent / "config.yaml")
    if args.data_dir is not None:
        config.data.dir = Path(args.data_dir).expanduser()

    regression_eval_dataset(
        config=config,
        split=args.split,
        ckpt_path=args.ckpt_path,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
