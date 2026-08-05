import numpy as np
import torch

from case_studies.cs1_cifar_label_noise.analyze import rank01
from case_studies.cs1_cifar_label_noise.model import resnet18_cifar
from case_studies.cs1_cifar_label_noise.train import inject_label_noise
from case_studies.quick_label_noise.run import make_data


def test_label_noise_artifact_is_exact_and_reproducible():
    original = np.arange(100, dtype=np.int64) % 10
    first_labels, first_mask = inject_label_noise(original, 0.17, np.random.default_rng(9))
    second_labels, second_mask = inject_label_noise(original, 0.17, np.random.default_rng(9))

    np.testing.assert_array_equal(first_labels, second_labels)
    np.testing.assert_array_equal(first_mask, second_mask)
    assert first_mask.dtype == np.bool_
    assert int(first_mask.sum()) == 17
    np.testing.assert_array_equal(first_labels[~first_mask], original[~first_mask])
    assert np.all(first_labels[first_mask] != original[first_mask])


def test_cifar_feature_cache_preserves_classifier_logits():
    model = resnet18_cifar().eval()
    inputs = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), model.fc(model.forward_features(inputs)))


def test_quick_label_noise_data_is_reproducible_and_exact():
    first = make_data(100, 0.2, 5)
    second = make_data(100, 0.2, 5)
    for left, right in zip(first[:4], second[:4], strict=True):
        torch.testing.assert_close(left, right)
    np.testing.assert_array_equal(first[4], second[4])
    assert int(first[4].sum()) == 16


def test_detector_rank_preserves_ties():
    ranks = rank01(np.array([-2.0, -2.0, 1.0, 4.0, 4.0]))
    np.testing.assert_allclose(ranks, [0.125, 0.125, 0.5, 0.875, 0.875])
