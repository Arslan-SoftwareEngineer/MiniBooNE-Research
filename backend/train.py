from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import robustness as rb

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "MiniBooNE_PID.csv"
OPENML_ID = "43974"
CLASSES = {
    0: "background (muon neutrino)",
    1: "signal (electron neutrino)",
}


def _isint(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _isnum(value):
    if isinstance(value, bool):
        return False
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def to_binary(values):
    s = pd.Series(values)
    out = []
    for v in s:
        if isinstance(v, (bool, np.bool_)):
            out.append(int(v))
            continue
        if _isint(v):
            out.append(int(v) if int(v) in (0, 1) else int(bool(v)))
            continue
        if _isnum(v):
            fv = float(v)
            out.append(int(bool(fv)))
            continue
        text = str(v).strip().lower()
        if text in {"true", "signal", "1", "yes", "event", "electron"}:
            out.append(1)
        elif text in {"false", "background", "0", "no", "noise", "muon"}:
            out.append(0)
        elif text.startswith("signal"):
            out.append(1)
        elif text.startswith("background"):
            out.append(0)
        else:
            raise ValueError(f"Unsupported label value: {v!r}")
    return np.asarray(out, dtype=int)


def _signal_label_candidates():
    return ["signal", "label", "target", "class", "y", "event"]


def check_positional_labels(y):
    labels = pd.Series(y).dropna()
    if labels.empty:
        return True
    values = set(str(v).strip().lower() for v in labels)
    if values.issubset({"0", "1", "true", "false", "signal", "background", "event", "background (muon neutrino)", "signal (electron neutrino)"}):
        return True
    return False


def _label_like(values):
    s = pd.Series(values)
    if s.empty:
        return False
    normalized = s.astype(str).str.strip().str.lower()
    if normalized.isin({"0", "1", "true", "false", "signal", "background"}).all():
        return True
    return False


def load_openml():
    """OpenML copy of MiniBooNE. Tries sklearn's fast pandas parser, then the slower liac-arff fallback."""
    from sklearn.datasets import fetch_openml

    try:
        bunch = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto")
    except TypeError:
        bunch = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="liac-arff")

    X = pd.DataFrame(bunch.data)
    y = pd.Series(bunch.target)
    return X, to_binary(y)


def load_uci(path: str | None = None):
    """Load the original UCI MiniBooNE file from an .txt file with the first line count header."""
    p = Path(path) if path is not None else (ROOT / "data" / "MiniBooNE_PID.txt")
    if not p.exists():
        raise FileNotFoundError(f"No UCI file found at {p}")

    rows = []
    labels = []
    with open(p, "r", encoding="utf-8", errors="replace") as handle:
        first_line = handle.readline()
        tokens = first_line.strip().split()
        if len(tokens) == 2 and all(_isnum(t) for t in tokens):
            n_signal, n_bkg = map(int, tokens)
            remaining_signal = n_signal
            remaining_bkg = n_bkg
            for line in handle:
                if not line.strip():
                    continue
                values = [float(v) for v in line.replace(",", " ").split()]
                if any(np.isclose(v, -999.0) for v in values):
                    continue
                rows.append(values)
                if remaining_signal > 0:
                    labels.append(1)
                    remaining_signal -= 1
                elif remaining_bkg > 0:
                    labels.append(0)
                    remaining_bkg -= 1
        else:
            # This is a plain data block without the original count header.
            row = [float(v) for v in first_line.replace(",", " ").split()]
            rows.append(row)
            for line in handle:
                if not line.strip():
                    continue
                row = [float(v) for v in line.replace(",", " ").split()]
                if any(np.isclose(v, -999.0) for v in row):
                    continue
                rows.append(row)
            labels = None

    if not rows:
        raise ValueError(f"No feature rows loaded from {p}")

    X = pd.DataFrame(np.asarray(rows, dtype=float))
    y = np.asarray(labels, dtype=int) if labels is not None else None
    if y is None:
        if X.shape[1] < 2:
            raise ValueError(f"Could not infer labels from UCI-style data file: {p}")
        y = to_binary(X.iloc[:, -1])
        X = X.iloc[:, :-1]
    return X, y


def load_csv(path: str, target: str | None = None):
    """Load a CSV file and detect a binary label column."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")

    def looks_like_headerless_file(path_obj: Path) -> bool:
        with open(path_obj, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                cells = [cell.strip() for cell in stripped.split(",")]
                cells = [cell for cell in cells if cell]
                if not cells:
                    continue
                return all(_isnum(cell) or cell.lower() in {"true", "false"} for cell in cells)
        return False

    reader = (
        lambda: pd.read_csv(p, header=None, na_values=["", "nan", "NaN"])
        if looks_like_headerless_file(p)
        else pd.read_csv(p, na_values=["", "nan", "NaN"])
    )

    try:
        df = reader()
    except Exception as exc:  # pragma: no cover - fallback for malformed CSVs
        raise SystemExit(f"Could not parse CSV: {p}") from exc

    if not isinstance(df, pd.DataFrame):
        raise SystemExit(f"CSV did not produce a table: {p}")

    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()
    if df.empty:
        raise SystemExit(f"CSV is empty: {p}")

    if target is not None:
        if target not in df.columns:
            raise SystemExit(f"Target column '{target}' not found in {p}. Use --target COLUMN_NAME.")
        y = to_binary(df[target])
        X = df.drop(columns=[target])
        return X, y

    if df.shape[0] > 0 and not looks_like_headerless_file(p):
        # Headered CSVs can still contain a true label-name column; drop the header row if it was kept as data.
        if not pd.to_numeric(df.iloc[0], errors="coerce").notna().all():
            df = df.iloc[1:].reset_index(drop=True)

    for candidate_name in _signal_label_candidates():
        for column in df.columns:
            if candidate_name in str(column).lower():
                X = df.drop(columns=[column])
                y = to_binary(df[column])
                if all(str(c).lstrip("-").replace(".", "", 1).isdigit() for c in X.columns):
                    X.columns = [f"ParticleID_{i}" for i in range(X.shape[1])]
                return X, y

    if df.shape[1] > 1 and _label_like(df.iloc[:, -1]):
        X = df.iloc[:, :-1]
        y = to_binary(df.iloc[:, -1])
        if all(str(c).lstrip("-").replace(".", "", 1).isdigit() for c in X.columns):
            X.columns = [f"ParticleID_{i}" for i in range(X.shape[1])]
        return X, y

    raise SystemExit(f"Could not detect a label column in {p}. Pass --target COLUMN_NAME.")

    if target is not None:
        if target not in df.columns:
            raise SystemExit(f"Target column '{target}' not found in {p}. Use --target COLUMN_NAME.")
        y = to_binary(df[target])
        X = df.drop(columns=[target])
        return X, y

    for candidate_name in _signal_label_candidates():
        for column in df.columns:
            if candidate_name in str(column).lower():
                X = df.drop(columns=[column])
                y = to_binary(df[column])
                if all(str(c).lstrip("-").replace(".", "", 1).isdigit() for c in X.columns):
                    X.columns = [f"ParticleID_{i}" for i in range(X.shape[1])]
                return X, y

    if df.shape[1] > 1 and _label_like(df.iloc[:, -1]):
        X = df.iloc[:, :-1]
        y = to_binary(df.iloc[:, -1])
        if all(str(c).lstrip("-").replace(".", "", 1).isdigit() for c in X.columns):
            X.columns = [f"ParticleID_{i}" for i in range(X.shape[1])]
        return X, y

    raise SystemExit(f"Could not detect a label column in {p}. Pass --target COLUMN_NAME.")


def load_file(path, target: str | None = None, signal_rows: int | None = None):
    """Read MiniBooNE from a local file and detect its layout automatically."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Could not load {p}. Put MiniBooNE_PID.csv in data/ or pass --source csv --csv PATH.")

    try:
        with open(p, "r", encoding="utf-8", errors="replace") as handle:
            first = handle.readline().strip()
    except Exception:
        first = ""

    if first and len(first.split()) == 2 and all(_isnum(v) for v in first.split()):
        X, y = load_uci(str(p))
        return X, y

    return load_csv(str(p), target=target)


def load_synthetic(n: int = 6000):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, 50)), columns=[f"f{i:02d}" for i in range(1, 51)])
    score = X["f01"] * 2.0 + X["f02"] * 1.5 + 0.7 * X["f05"] + 0.4 * X["f10"]
    y = (score + rng.normal(scale=0.7, size=n) > 0.5).astype(int)
    return X, y


def fit_models(X_train, y_train, quick: bool = False, with_mlp: bool = True, seed: int = 42, augment_copies: int = 1):
    """Fit baseline XGB, dropout-robust XGB and (optionally) an MLP."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    models = {}

    baseline = rb.make_xgb(quick=quick, seed=seed)
    baseline.fit(X_train, y_train)
    models["baseline"] = baseline

    rng = np.random.default_rng(seed + 10)
    X_aug, y_aug = rb.augment_with_dropout(X_train, y_train, rng, copies=augment_copies)
    robust = rb.make_xgb(quick=quick, seed=seed + 1)
    robust.fit(X_aug, y_aug)
    models["robust"] = robust

    if with_mlp:
        mlp = rb.make_mlp(quick=quick, seed=seed + 2)
        mlp.fit(X_train, y_train)
        models["mlp"] = mlp

    return models


def build_bundle(models, features, model_id, synthetic: bool):
    bundle = {
        "classes": {str(k): v for k, v in CLASSES.items()},
        "features": list(features),
        "model_id": model_id,
        "model_version": "1.0",
        "models": models,
        "pipeline": "xgb+robustness",
        "synthetic": bool(synthetic),
        "task": "miniboone_robustness",
    }
    return bundle


def build_model_info(models, X, features, n_train=None, n_test=None, synthetic=False):
    X_arr = X.to_numpy(float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    baseline = models.get("baseline")
    if baseline is None:
        raise ValueError("A baseline model is required to build feature metadata.")
    model = baseline.named_steps["model"] if hasattr(baseline, "named_steps") else baseline
    importance = np.asarray(model.feature_importances_, dtype=float)
    order = np.argsort(importance)[::-1]
    feature_rows = []
    for idx, pos in enumerate(order):
        name = features[pos]
        col = X_arr[:, pos]
        feature_rows.append(
            {
                "name": name,
                "min": float(col.min()),
                "max": float(col.max()),
                "mean": float(col.mean()),
                "std": float(col.std(ddof=0) if col.size > 1 else 0.0),
                "importance": float(importance[pos]),
                "rank": int(idx + 1),
            }
        )
    model_info = {
        "model_id": str(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")),
        "synthetic": bool(synthetic),
        "classes": {str(k): v for k, v in CLASSES.items()},
        "n_train": int(n_train) if n_train is not None else None,
        "n_test": int(n_test) if n_test is not None else None,
        "features": feature_rows,
        "version": "1.0",
    }
    return model_info


def save_figures(results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    rates = results.get("dropout", {}).get("rates", [])
    series = results.get("dropout", {}).get("series", {})
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, info in series.items():
        ax.plot(rates, info.get("mean", []), marker="o", label=name)
    ax.set_xlabel("dropout rate")
    ax.set_ylabel("accuracy")
    ax.set_title("Dropout robustness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "dropout.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for group, payload in results.get("noise", {}).get("groups", {}).items():
        for name, info in payload.get("series", {}).items():
            ax.plot(results.get("noise", {}).get("alphas", []), info.get("mean", []), marker="o", label=f"{group}:{name}")
    ax.set_xlabel("noise scale")
    ax.set_ylabel("accuracy")
    ax.set_title("Noise robustness")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "noise.png", dpi=150)
    plt.close(fig)


def load_auto(target=None, signal_rows=None):
    if DEFAULT_DATA.exists():
        return load_file(DEFAULT_DATA, target=target, signal_rows=signal_rows)

    try:
        return load_openml()
    except Exception:
        pass

    try:
        return load_uci(str(ROOT / "data" / "MiniBooNE_PID.txt"))
    except Exception:
        raise SystemExit(
            "Could not load MiniBooNE automatically. Put data/MiniBooNE_PID.csv in the repo root or run with --source uci --csv PATH."
        )


def _write_artifacts(bundle, model_info, results, feature_df, artifacts_dir: Path, synthetic: bool):
    artifacts_dir = Path(artifacts_dir)
    models_dir = artifacts_dir / "models"
    public_dir = artifacts_dir / "public"
    models_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, models_dir / "model_bundle.joblib")
    (public_dir / "model_info.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")
    (public_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    root_docs = ROOT / "docs"
    root_docs.mkdir(exist_ok=True)
    (ROOT / "docs" / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (ROOT / "frontend" / "public" / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (ROOT / "frontend" / "public" / "model_info.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    if feature_df is not None:
        feature_df.to_csv(ROOT / "data" / "sample_inputs.csv", index=False)

    save_figures(results, ROOT / "docs" / "figures")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train MiniBooNE robustness models")
    parser.add_argument("--source", choices=["auto", "openml", "uci", "csv", "synthetic"], default="auto")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-mlp", action="store_true")
    parser.add_argument("--artifacts-dir", type=str, default=str(ROOT / "artifacts"))
    args = parser.parse_args(argv)

    if args.source == "synthetic":
        X_df, y = load_synthetic(n=6000 if not args.quick else 1000)
        synthetic = True
    elif args.source == "csv":
        if not args.csv:
            raise SystemExit("--csv PATH is required when --source csv is used.")
        X_df, y = load_file(args.csv, target=args.target)
        synthetic = False
    elif args.source == "uci":
        path = args.csv or str(ROOT / "data" / "MiniBooNE_PID.txt")
        X_df, y = load_uci(path)
        synthetic = False
    elif args.source == "openml":
        X_df, y = load_openml()
        synthetic = False
    else:
        X_df, y = load_auto(target=args.target)
        synthetic = False

    X_df = X_df.reset_index(drop=True)
    y = np.asarray(y, dtype=int)
    if len(X_df) != len(y):
        raise ValueError("Feature table and labels have mismatched lengths")

    X_train, X_test, y_train, y_test = train_test_split(X_df.to_numpy(float), y, test_size=0.2, random_state=42, stratify=y)
    models = fit_models(X_train, y_train, quick=args.quick, with_mlp=not args.no_mlp, seed=42, augment_copies=1)

    model_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle = build_bundle(models, list(X_df.columns), model_id, synthetic)
    model_info = build_model_info(models, X_df, list(X_df.columns), n_train=len(X_df), n_test=len(X_test), synthetic=synthetic)

    rates = [0.1, 0.2, 0.3, 0.4]
    noise_alphas = [0.0, 0.5, 1.0, 1.5, 2.0]
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    importance = models["baseline"].named_steps["model"].feature_importances_
    results = {
        "model_id": model_id,
        "synthetic": synthetic,
        "dropout": rb.dropout_experiment(models, X_test, y_test, X_train_mean, rates, seeds=3, impute_key="baseline"),
        "noise": rb.noise_experiment(models, X_test, y_test, X_train_std, importance, noise_alphas, seeds=3, k=5),
        "replication": rb.replication_table(models, X_test, y_test),
    }

    _write_artifacts(bundle, model_info, results, X_df.head(1), Path(args.artifacts_dir), synthetic)

    print(f"Model trained: {model_id}")
    print(f"XGBoost: {rb.XGBClassifier().get_xgb_params().get('random_state', 'unknown')}")
    print(f"scikit-learn version: {__import__('sklearn').__version__}")
    print(f"Saved bundle to {Path(args.artifacts_dir) / 'models' / 'model_bundle.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
