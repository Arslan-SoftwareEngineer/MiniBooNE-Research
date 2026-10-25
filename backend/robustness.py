"""Corruption helpers, model fitting and the robustness experiments.

Everything works on plain numpy arrays so that training and inference see the
same input type (feature names are stored separately in the model bundle).

Sensor failure is encoded as 0.0 in RAW feature space (the same encoding the
app sends for an "offline" sensor).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# --------------------------------------------------------------------------
# Corruptions
# --------------------------------------------------------------------------


def n_dropped(n_features: int, rate: float) -> int:
    return max(1, int(round(rate * n_features)))


def random_columns(n_features: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Pick a random subset of sensors (the same subset for a whole test set)."""
    return rng.choice(n_features, size=n_dropped(n_features, rate), replace=False)


def drop_columns(X: np.ndarray, cols) -> np.ndarray:
    """Return a copy of X where the given sensor columns read 0 (dead sensor)."""
    Xc = np.array(X, dtype=float, copy=True)
    Xc[:, np.asarray(cols, dtype=int)] = 0.0
    return Xc


def impute_columns(X: np.ndarray, cols, fill: np.ndarray) -> np.ndarray:
    """Replace the given columns by training means (needs to know which sensors died)."""
    Xc = np.array(X, dtype=float, copy=True)
    cols = np.asarray(cols, dtype=int)
    Xc[:, cols] = fill[cols]
    return Xc


def add_noise(X: np.ndarray, cols, std: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise (sigma = alpha * training std of that feature) to given columns."""
    Xc = np.array(X, dtype=float, copy=True)
    cols = np.asarray(cols, dtype=int)
    Xc[:, cols] += rng.normal(size=(len(Xc), len(cols))) * (alpha * std[cols])
    return Xc


def augment_with_dropout(X, y, rng, rate_range=(0.1, 0.4), copies=1):
    """Clean data plus `copies` corrupted copies.

    Every corrupted row gets its OWN random set of dead sensors, with a dropout
    rate drawn uniformly from `rate_range`, so the model sees many failure
    patterns (at test time one fixed pattern applies to a whole batch).
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    xs, ys = [X], [np.asarray(y)]
    for _ in range(copies):
        k = np.rint(rng.uniform(*rate_range, size=n) * d).astype(int)
        k = np.clip(k, 1, d)
        ranks = rng.random((n, d)).argsort(axis=1).argsort(axis=1)
        mask = ranks < k[:, None]
        Xc = X.copy()
        Xc[mask] = 0.0
        xs.append(Xc)
        ys.append(np.asarray(y))
    return np.vstack(xs), np.concatenate(ys)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def make_xgb(quick: bool = False, seed: int = 42) -> Pipeline:
    clf = XGBClassifier(
        n_estimators=60 if quick else 400,
        max_depth=4 if quick else 6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        importance_type="gain",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=seed,
    )
    return Pipeline([("model", clf)])


def make_mlp(quick: bool = False, seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64,) if quick else (128, 64),
                    early_stopping=True,
                    max_iter=30 if quick else 100,
                    random_state=seed,
                ),
            ),
        ]
    )


def feature_importance(pipe: Pipeline) -> np.ndarray:
    return np.asarray(pipe.named_steps["model"].feature_importances_, dtype=float)


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------


def _acc(model, X, y) -> float:
    return float(accuracy_score(y, model.predict(X)))


def _mean_std(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std())}


def dropout_experiment(models: dict, X_test, y_test, X_train_mean, rates, seeds, impute_key="xgb_baseline"):
    """Accuracy when a random subset of sensors reads 0, for each model.

    Adds an extra series `<impute_key>_mean_impute`: same baseline model, but the
    dead sensors are replaced by training means (this assumes the system KNOWS
    which sensors died, so it is an optimistic reference, not a fair rival).
    """
    d = X_test.shape[1]
    names = list(models) + [f"{impute_key}_mean_impute"]
    acc = {n: {r: [] for r in rates} for n in names}
    for r in rates:
        for s in range(seeds):
            rng = np.random.default_rng(1000 + s)
            cols = random_columns(d, r, rng) if r > 0 else np.array([], dtype=int)
            Xd = drop_columns(X_test, cols) if r > 0 else X_test
            for name, m in models.items():
                acc[name][r].append(_acc(m, Xd, y_test))
            Xi = impute_columns(X_test, cols, X_train_mean) if r > 0 else X_test
            acc[f"{impute_key}_mean_impute"][r].append(_acc(models[impute_key], Xi, y_test))
    return {
        "rates": list(rates),
        "seeds": seeds,
        "series": {n: {"mean": [_mean_std(acc[n][r])["mean"] for r in rates],
                       "std": [_mean_std(acc[n][r])["std"] for r in rates]} for n in names},
    }


def noise_experiment(models: dict, X_test, y_test, X_train_std, importance, alphas, seeds, k=5):
    """Gaussian noise on the k most vs k least important sensors."""
    order = np.argsort(importance)[::-1]
    groups = {"top": order[:k], "bottom": order[-k:]}
    out = {"alphas": list(alphas), "seeds": seeds, "k": k, "groups": {}}
    for gname, cols in groups.items():
        series = {}
        for name, m in models.items():
            means, stds = [], []
            for a in alphas:
                vals = []
                for s in range(seeds):
                    rng = np.random.default_rng(2000 + s)
                    Xn = add_noise(X_test, cols, X_train_std, a, rng) if a > 0 else X_test
                    vals.append(_acc(m, Xn, y_test))
                ms = _mean_std(vals)
                means.append(ms["mean"])
                stds.append(ms["std"])
            series[name] = {"mean": means, "std": stds}
        out["groups"][gname] = {"columns": [int(c) for c in cols], "series": series}
    return out


def collapse_threshold(alphas, means, drop_points: float = 0.05):
    """Smallest noise level at which accuracy is `drop_points` below the clean value."""
    base = means[0]
    for a, m in zip(alphas, means):
        if base - m >= drop_points:
            return float(a)
    return None


def replication_table(models: dict, X_test, y_test) -> list[dict]:
    rows = []
    for name, m in models.items():
        proba = m.predict_proba(X_test)[:, 1]
        rows.append({
            "model": name,
            "accuracy": float(accuracy_score(y_test, (proba >= 0.5).astype(int))),
            "auc": float(roc_auc_score(y_test, proba)),
        })
    return rows
