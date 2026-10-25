import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import train  # noqa: E402


@pytest.fixture(scope="session")
def bundle_path(tmp_path_factory):
    """Train tiny synthetic models once, so CI needs no committed model file."""
    X_df, y = train.load_synthetic(n=1500)
    X = X_df.to_numpy(float)
    models = train.fit_models(X, y, quick=True, with_mlp=False)
    bundle = train.build_bundle(models, X_df.columns, "test-model", synthetic=True)
    path = tmp_path_factory.mktemp("models") / "model_bundle.joblib"
    joblib.dump(bundle, path)
    return path


@pytest.fixture(scope="session")
def client(bundle_path):
    from app import create_app

    return create_app(bundle_path).test_client()


@pytest.fixture(scope="session")
def features(bundle_path):
    return joblib.load(bundle_path)["features"]


@pytest.fixture()
def payload(features):
    rng = np.random.default_rng(0)
    return {f: float(v) for f, v in zip(features, rng.normal(size=len(features)))}
