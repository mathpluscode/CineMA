"""Script to eval any wandb folder."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from cinema.classification.eval import classification_eval_dataset as cls_eval
from cinema.log import get_logger
from cinema.regression.eval import regression_eval_dataset as reg_eval
from cinema.regression.landmark.eval import landmark_regression_eval_dataset as landmark_reg_eval
from cinema.segmentation.emidec.eval import segmentation_eval_emidec_dataset as emidec_seg_eval
from cinema.segmentation.eval import segmentation_eval_edes_dataset as edes_seg_eval
from cinema.segmentation.kaggle.eval import segmentation_eval_kaggle_dataset as kaggle_seg_eval
from cinema.segmentation.landmark.eval import landmark_detection_eval_dataset as landmark_eval
from cinema.segmentation.myops2020.eval import segmentation_eval_myops2020_dataset as myops2020_seg_eval
from cinema.segmentation.rescan.ef_eval import ef_eval_rescan_dataset as rescan_seg_ef_eval
from cinema.segmentation.rescan.eval import segmentation_eval_rescan_dataset as rescan_seg_eval

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder_path",
        type=Path,
        help="Path to the wandb folder.",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Which dataset to eval.",
        default="",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "test_retest_100"],
        help="Split of data, train or test. test_retest_100 is for rescan",
        default="test",
    )
    parser.add_argument("--save", action="store_true", help="Save the predictions.")
    parser.set_defaults(save=False)
    args = parser.parse_args()

    return args


def main() -> None:  # noqa: C901
    """Main function."""
    args = parse_args()

    config_path = args.folder_path / "ckpt" / "config.yaml"
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return
    config = OmegaConf.load(config_path)
    logger.info(f"Using config: {config_path}")

    # update dataset
    data = args.data if args.data else config.data.name
    # .../acdc/processed
    config.data.dir = Path(config.data.dir).parent.parent / data / "processed"
    logger.info(f"Using data: {config.data.dir}")

    # find the last checkpoint
    ckpt_paths = list(args.folder_path.glob("ckpt/*.pt"))
    if len(ckpt_paths) == 0:
        logger.error("No checkpoint found.")
        return
    ckpt_path = max(ckpt_paths, key=lambda x: int(x.stem.split("_")[-1]))
    logger.info(f"Using checkpoint: {ckpt_path}")

    # update output directory
    out_dir = ckpt_path.parent / f"{data}_eval_{ckpt_path.stem}"
    out_dir.mkdir(exist_ok=True)

    if config.task == "classification":
        if data in ["acdc", "mnms", "mnms2", "emidec"]:
            cls_eval(
                config=config,
                split=args.split,
                ckpt_path=ckpt_path,
                out_dir=out_dir,
            )
        else:
            raise ValueError(f"Unknown dataset: {data}")
    elif config.task == "regression":
        if data in ["acdc", "mnms", "mnms2", "emidec"]:
            reg_eval(
                config=config,
                split=args.split,
                ckpt_path=ckpt_path,
                out_dir=out_dir,
            )
        elif data == "landmark":
            landmark_reg_eval(
                config=config,
                split=args.split,
                ckpt_path=ckpt_path,
                out_dir=out_dir,
                save=args.save,
            )
        else:
            raise ValueError(f"Unknown dataset: {data}")
    elif config.task == "segmentation":
        if data == "rescan":
            if args.split == "test_retest_100":
                rescan_seg_ef_eval(
                    config=config,
                    split=args.split,
                    ckpt_path=ckpt_path,
                    out_dir=out_dir,
                    save=args.save,
                )
            else:
                rescan_seg_eval(
                    config=config,
                    split=args.split,
                    ckpt_path=ckpt_path,
                    out_dir=out_dir,
                    save=args.save,
                )
        elif data == "emidec":
            emidec_seg_eval(
                config=config,
                split=args.split,
                ckpt_path=ckpt_path,
                out_dir=out_dir,
                save=args.save,
            )
        elif data == "myops2020":
            myops2020_seg_eval(
                config=config,
                split=args.split,
                ckpt_path=ckpt_path,
                out_dir=out_dir,
                save=args.save,
            )
        elif data in ["acdc", "mnms", "mnms2"]:
            edes_seg_eval(
                config=config,
                split=args.split,
                ckpt_path=ckpt_path,
                out_dir=out_dir,
                save=args.save,
            )
        elif data == "kaggle":
            kaggle_seg_eval(
                config=config,
                split=args.split,
                ckpt_path=ckpt_path,
                out_dir=out_dir,
                save=args.save,
            )
        elif data == "landmark":
            landmark_eval(
                config=config,
                split=args.split,
                ckpt_path=ckpt_path,
                out_dir=out_dir,
                save=args.save,
            )
        else:
            raise ValueError(f"Unknown dataset: {data}")
    else:
        raise ValueError(f"Unknown evaluation task: {config.task}")


if __name__ == "__main__":
    main()
