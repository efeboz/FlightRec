import numpy as np

from case_studies.cs1_cifar_label_noise.train import inject_label_noise


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
