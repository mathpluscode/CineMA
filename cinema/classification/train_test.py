"""Tests for functions in training."""

import numpy as np
import pytest

from cinema.classification.train import classification_metrics


@pytest.mark.parametrize("n_samples", [1, 2, 10, 100, 1000])
@pytest.mark.parametrize("n_classes", [2, 3, 4, 100])
def test_multiclass_classification_eval_metrics(n_samples: int, n_classes: int) -> None:
    """Test output shapes.

    Args:
        n_samples: number of samples.
        n_classes: number of classes.
    """
    rng = np.random.default_rng()
    true_labels = rng.choice(n_classes, n_samples)
    pred_labels = rng.choice(n_classes, n_samples)
    pred_probs = rng.random((n_samples, n_classes))
    pred_probs /= pred_probs.sum(axis=1, keepdims=True)
    metrics = classification_metrics(
        true_labels=true_labels,
        pred_labels=pred_labels,
        pred_probs=pred_probs,
    )
    for metric in metrics.values():
        assert isinstance(metric, float)
