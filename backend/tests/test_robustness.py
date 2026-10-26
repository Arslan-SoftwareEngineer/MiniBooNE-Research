import numpy as np

import robustness as rb


def test_drop_columns_zeroes_only_selected():
    X = np.ones((10, 6))
    out = rb.drop_columns(X, [1, 4])
    assert (out[:, [1, 4]] == 0).all()
    assert (out[:, [0, 2, 3, 5]] == 1).all()
    assert (X == 1).all(), "input must not be modified"


def test_random_columns_size_and_uniqueness():
    cols = rb.random_columns(50, 0.3, np.random.default_rng(0))
    assert len(cols) == 15 and len(set(cols)) == 15


def test_augment_with_dropout_shapes_and_rates():
    X = np.ones((400, 50))
    y = np.arange(400) % 2
    Xa, ya = rb.augment_with_dropout(X, y, np.random.default_rng(0), (0.1, 0.4), copies=2)
    assert Xa.shape == (1200, 50) and ya.shape == (1200,)
    assert (Xa[:400] == 1).all(), "first block is the untouched clean data"
    zero_frac = (Xa[400:] == 0).mean(axis=1)
    assert zero_frac.min() >= 0.1 - 1 / 50 and zero_frac.max() <= 0.4 + 1 / 50


def test_add_noise_only_selected_columns():
    X = np.zeros((100, 5))
    out = rb.add_noise(X, [0, 1], np.ones(5), 1.0, np.random.default_rng(0))
    assert out[:, :2].std() > 0.5 and (out[:, 2:] == 0).all()


def test_collapse_threshold():
    assert rb.collapse_threshold([0, 1, 2], [0.9, 0.88, 0.80]) == 2.0
    assert rb.collapse_threshold([0, 1, 2], [0.9, 0.89, 0.88]) is None
