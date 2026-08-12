import numpy as np
import torch

from case_studies.cs1_cifar_label_noise.analyze import rank01
from case_studies.cs1_cifar_label_noise.model import resnet18_cifar
from case_studies.cs1_cifar_label_noise.train import inject_label_noise
from case_studies.mid_digits_noise.train import DigitCNN, load_digit_data
from case_studies.mid_digits_noise.train import inject_label_noise as inject_digit_noise
from case_studies.quick_influence_sanity.run import audit_gradients
from case_studies.quick_influence_sanity.run import make_data as make_influence_data
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


def test_quick_influence_data_is_reproducible_and_standardized():
    features, labels, mask = make_influence_data(120, 8, 13)
    repeated = make_influence_data(120, 8, 13)
    torch.testing.assert_close(features, repeated[0])
    torch.testing.assert_close(labels, repeated[1])
    np.testing.assert_array_equal(mask, repeated[2])
    assert int(mask.sum()) == 8
    torch.testing.assert_close(
        features.mean(0), torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0
    )


def test_quick_influence_gradient_audit_flags_broken_code():
    reports = audit_gradients()
    correct = reports["correct_custom_function"]
    assert correct.passed
    assert correct.max_abs_err_cs is not None and correct.max_abs_err_cs < 1e-12
    broken = reports["broken_backward"]
    assert not broken.passed
    assert broken.max_abs_err_fd > 0.1
    assert broken.max_abs_err_cs is not None and broken.max_abs_err_cs > 0.1
    detached = reports["detached_branch"]
    assert not detached.passed
    assert set(detached.unreachable_params) == {"orphan.weight", "orphan.bias"}


def test_digit_noise_injection_is_exact_and_reproducible():
    original = np.arange(200, dtype=np.int64) % 10
    labels, mask = inject_digit_noise(original, 0.1, np.random.default_rng(3))
    repeated, repeated_mask = inject_digit_noise(original, 0.1, np.random.default_rng(3))

    np.testing.assert_array_equal(labels, repeated)
    np.testing.assert_array_equal(mask, repeated_mask)
    assert int(mask.sum()) == 20
    np.testing.assert_array_equal(labels[~mask], original[~mask])
    assert np.all(labels[mask] != original[mask])
    assert labels.min() >= 0 and labels.max() <= 9


def test_digit_data_split_is_stratified_and_scaled():
    train_x, train_y, test_x, test_y = load_digit_data(0)
    assert train_x.shape[1:] == (1, 8, 8)
    assert len(train_x) == len(train_y) == 1437
    assert len(test_x) == len(test_y) == 360
    assert 0.0 <= float(train_x.min()) and float(train_x.max()) <= 1.0
    counts = torch.bincount(train_y, minlength=10)
    assert int(counts.min()) > 0
    assert int(counts.max() - counts.min()) <= 10  # stratification keeps classes balanced
    torch.testing.assert_close(train_x, load_digit_data(0)[0])


def test_digit_model_head_is_the_last_layer_influence_selects():
    model = DigitCNN()
    names = [name for name, _ in model.named_parameters()]
    assert names[-2:] == ["head.weight", "head.bias"]
    with torch.no_grad():
        assert model(torch.zeros(2, 1, 8, 8)).shape == (2, 10)
