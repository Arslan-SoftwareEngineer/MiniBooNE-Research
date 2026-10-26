"""Data-loader fallbacks (no network needed: everything is mocked)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import train


def _fake_bunch(n=6):
    X = pd.DataFrame(np.arange(n * 3, dtype=float).reshape(n, 3), columns=["a", "b", "c"])
    y = pd.Series(["True", "False"] * (n // 2))
    return SimpleNamespace(data=X, target=y)


def test_openml_falls_back_to_liac_arff_parser(monkeypatch):
    calls = []

    def fake_fetch(**kw):
        calls.append(kw["parser"])
        if kw["parser"] == "auto":
            raise TypeError("expected string or bytes-like object, got 'bool'")  # the real-world crash
        return _fake_bunch()

    monkeypatch.setattr("sklearn.datasets.fetch_openml", fake_fetch)
    X, y = train.load_openml()
    assert calls == ["auto", "liac-arff"]
    assert X.shape == (6, 3) and y.tolist() == [1, 0, 1, 0, 1, 0]


def _write_uci(path, rows, n_sig, n_bkg):
    with open(path, "w") as f:
        f.write(f" {n_sig} {n_bkg}\n")
        for r in rows:
            f.write("  ".join(f"{v:.5e}" for v in r) + "\n")


def test_uci_parser_labels_and_missing_code(tmp_path):
    rows = np.random.default_rng(0).normal(size=(7, 50))
    rows[3, 5] = -999  # missing-value code: that row must be dropped
    p = tmp_path / "uci.txt"
    _write_uci(p, rows, 3, 4)
    X, y = train.load_uci(str(p))
    assert X.shape == (6, 50) and y.tolist() == [1, 1, 1, 0, 0, 0]


def test_auto_falls_back_to_uci_when_openml_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "DEFAULT_DATA", tmp_path / "absent.csv")
    def boom(**kw):
        raise OSError("getaddrinfo failed")

    monkeypatch.setattr("sklearn.datasets.fetch_openml", boom)
    p = tmp_path / "uci.txt"
    _write_uci(p, np.random.default_rng(1).normal(size=(5, 50)), 2, 3)
    monkeypatch.setattr(train, "load_uci", lambda path=None, _orig=train.load_uci: _orig(str(p)))
    X, y = train.load_auto()
    assert X.shape == (5, 50) and y.sum() == 2


def test_auto_gives_actionable_message_when_everything_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "DEFAULT_DATA", tmp_path / "absent.csv")
    def boom(**kw):
        raise OSError("getaddrinfo failed")

    monkeypatch.setattr("sklearn.datasets.fetch_openml", boom)
    monkeypatch.setattr(train, "load_uci", lambda path=None: (_ for _ in ()).throw(OSError("no network")))
    with pytest.raises(SystemExit) as e:
        train.load_auto()
    assert "--source uci" in str(e.value) and "MiniBooNE_PID.csv" in str(e.value)


# ---- local-file layouts (the user's data/MiniBooNE_PID.csv can be any of these) -------------
def _frame(n=8, d=50, seed=0):
    return pd.DataFrame(np.random.default_rng(seed).normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])


def test_file_uci_layout_with_csv_extension(tmp_path):
    p = tmp_path / "MiniBooNE_PID.csv"
    _write_uci(p, np.random.default_rng(0).normal(size=(7, 50)), 3, 4)
    X, y = train.load_file(p)
    assert X.shape == (7, 50) and y.tolist() == [1, 1, 1, 0, 0, 0, 0]


def test_file_header_with_boolean_signal_column(tmp_path):
    df = _frame()
    df["signal"] = [True, False] * 4
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    X, y = train.load_file(p)
    assert X.shape == (8, 50) and "signal" not in X.columns and y.tolist() == [1, 0] * 4


def test_file_header_numeric_label_last_column_and_index_column(tmp_path):
    df = _frame()
    df["whatever"] = [1, 0, 0, 1, 1, 0, 1, 0]
    p = tmp_path / "d.csv"
    df.to_csv(p)  # writes an unnamed index column too
    X, y = train.load_file(p)
    assert X.shape == (8, 50) and y.tolist() == [1, 0, 0, 1, 1, 0, 1, 0]


def test_file_without_header_label_last(tmp_path):
    df = _frame()
    df["y"] = [0, 1] * 4
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False, header=False)
    X, y = train.load_file(p)
    assert X.shape == (8, 50) and list(X.columns[:2]) == ["ParticleID_0", "ParticleID_1"] and y.tolist() == [0, 1] * 4


def test_file_explicit_target_and_text_labels(tmp_path):
    df = _frame()
    df["kind"] = ["signal", "background"] * 4
    df["other"] = np.arange(8)
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    X, y = train.load_file(p, target="kind")
    assert y.tolist() == [1, 0] * 4 and "other" in X.columns


def test_file_missing_label_is_reported(tmp_path):
    p = tmp_path / "d.csv"
    _frame().to_csv(p, index=False)
    with pytest.raises(SystemExit) as e:
        train.load_file(p)
    assert "--target" in str(e.value)


def test_file_not_found_message(tmp_path):
    with pytest.raises(SystemExit) as e:
        train.load_file(tmp_path / "nope.csv")
    assert "MiniBooNE_PID.csv" in str(e.value)
