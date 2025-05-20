"""Test all scripts."""

import pytest
import torch

from cinema.examples.inference.classification_cvd import run as run_clf_cvd
from cinema.examples.inference.classification_sex import run as run_clf_sex
from cinema.examples.inference.classification_vendor import run as run_clf_vendor
from cinema.examples.inference.landmark_coordinate import run as run_landmark_coordinate
from cinema.examples.inference.landmark_heatmap import run as run_landmark_heatmap
from cinema.examples.inference.mae import run as run_mae
from cinema.examples.inference.mae_feature_extraction import run as run_mae_feature_extraction
from cinema.examples.inference.regression_age import run as run_regression_age
from cinema.examples.inference.regression_bmi import run as run_regression_bmi
from cinema.examples.inference.regression_ef import run as run_regression_ef
from cinema.examples.inference.segmentation_lax_4c import run as run_segmentation_lax_4c
from cinema.examples.inference.segmentation_sax import run as run_segmentation_sax


@pytest.mark.integration
def test_mae_scripts() -> None:
    """Execute all scripts."""
    run_mae("cpu", torch.float32)
    run_mae_feature_extraction("cpu", torch.float32)


@pytest.mark.integration
def test_classifcation_scripts() -> None:
    """Execute all scripts."""
    run_clf_cvd("acdc", "sax", 0, "cpu", torch.float32)
    run_clf_cvd("mnms2", "lax_4c", 0, "cpu", torch.float32)
    run_clf_sex(0, "cpu", torch.float32)
    run_clf_vendor("sax", 0, "cpu", torch.float32)
    run_clf_vendor("lax_4c", 0, "cpu", torch.float32)


@pytest.mark.integration
def test_regression_scripts() -> None:
    """Execute all scripts."""
    run_regression_age(0, "cpu", torch.float32)
    run_regression_bmi(0, "cpu", torch.float32)
    run_regression_ef("mnms", "sax", 0, "cpu", torch.float32)
    run_regression_ef("mnms2", "sax", 0, "cpu", torch.float32)


@pytest.mark.integration
def test_landmark_scripts() -> None:
    """Execute all scripts."""
    run_landmark_coordinate("lax_2c", 0, "cpu", torch.float32)
    run_landmark_heatmap("lax_2c", 0, "cpu", torch.float32)


@pytest.mark.integration
def test_segmentation_scripts() -> None:
    """Execute all scripts."""
    run_segmentation_lax_4c("mnms2", 0, "cpu", torch.float32)
    run_segmentation_sax("mnms2", 0, "cpu", torch.float32)
