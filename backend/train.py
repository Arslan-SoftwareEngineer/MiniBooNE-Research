"""Train baseline + dropout-robust models on MiniBooNE and run the robustness study.

Examples (run from the project root, venv active):

    python backend/train.py                      # real data from OpenML (needs internet)
    python backend/train.py --source uci         # original UCI file instead
    python backend/train.py --source csv --csv data/miniboone.csv --target signal
    python backend/train.py --source synthetic --quick   # 20-second smoke test, NOT real results

Outputs
    backend/models/model_bundle.joblib      model bundle used by the Flask API
    backend/test_payload.json               one valid request body (manual stage 5)
    data/sample_inputs.csv                  a few non-sensitive sample rows (manual stage 5)
    frontend/public/model_info.json         feature schema + samples for the Angular form
    frontend/public/results.json            replication + robustness results for the Research tab
    docs/results.json, docs/figures/*.png   same results for the technical report
"""
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robustness as rb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt"
OPENML_ID = 41150  # "MiniBooNE" on OpenML: 130,064 rows, 50 features, target = signal
CLASSES = {"0": "background (muon neutrino)", "1": "signal (electron neutrino)"}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_openml():
    from sklearn.datasets import fetch_openml

    ds = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto", data_home=str(ROOT / "data" / "openml_cache"))
    X = ds.data.astype(float)
    y = ds.target.astype(str).str.lower().isin(["true", "1", "signal"]).astype(int).to_numpy()
    return X, y


def load_uci(path: str | None = None):
    import urllib.request

    target = Path(path) if path else ROOT / "data" / "MiniBooNE_PID.txt"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {UCI_URL} ...")
        urllib.request.urlretrieve(UCI_URL, target)
    with open(target) as f:
        n_sig, n_bkg = (int(v) for v in f.readline().split())
    df = pd.read_csv(target, sep=r"\s+", skiprows=1, header=None, engine="python")
    df = df.iloc[: n_sig + n_bkg]
    y = np.r_[np.ones(n_sig, dtype=int), np.zeros(n_bkg, dtype=int)]
    X = df.astype(float)
    X.columns = [f"ParticleID_{i}" for i in range(X.shape[1])]
    bad = (X == -999).any(axis=1).to_numpy()
    if bad.any():
        print(f"Dropping {bad.sum()} rows containing the -999 'missing' code")
        X, y = X[~bad].reset_index(drop=True), y[~bad]
    return X, y


def load_csv(path: str, target: str):
    df = pd.read_csv(path)
    y = df[target]
    y = y.astype(str).str.lower().isin(["true", "1", "signal", "1.0"]).astype(int).to_numpy()
    return df.drop(columns=[target]).astype(float), y


def load_synthetic(n: int = 6000):
    from sklearn.datasets import make_classification

    Xa, y = make_classification(n_samples=n, n_features=50, n_informative=15, n_redundant=10,
                                class_sep=1.0, flip_y=0.03, random_state=0)
    return pd.DataFrame(Xa, columns=[f"f{i + 1:02d}" for i in range(50)]), y


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def fit_models(X_train, y_train, quick=False, with_mlp=True, seed=42, augment_copies=1):
    """Fit baseline XGB, dropout-robust XGB and (optionally) an MLP."""
    models = {}
    t = time.time()
    models["xgb_baseline"] = rb.make_xgb(quick, seed).fit(X_train, y_train)
    print(f"  baseline XGBoost fitted in {time.time() - t:.1f}s")

    t = time.time()
    rng = np.random.default_rng(seed)
    Xa, ya = rb.augment_with_dropout(X_train, y_train, rng, rate_range=(0.1, 0.4), copies=augment_copies)
    models["xgb_robust"] = rb.make_xgb(quick, seed).fit(Xa, ya)
    print(f"  dropout-augmented XGBoost fitted on {len(Xa)} rows in {time.time() - t:.1f}s")

    if with_mlp:
        t = time.time()
        models["mlp"] = rb.make_mlp(quick, seed).fit(X_train, y_train)
        print(f"  MLP fitted in {time.time() - t:.1f}s")
    return models


def build_bundle(models, features, model_id, synthetic):
    return {
        "pipeline": models["xgb_baseline"],  # manual-compatible default
        "models": {"baseline": models["xgb_baseline"], "robust": models["xgb_robust"]},
        "features": list(features),
        "task": "classification",
        "classes": CLASSES,
        "model_version": "1.0",
        "model_id": model_id,
        "synthetic": bool(synthetic),
    }


def r6(x):
    return float(f"{float(x):.6g}")


def build_model_info(X_train_df, importance, X_test, y_test, models, model_id, synthetic, n_train, n_test):
    feats = list(X_train_df.columns)
    ranks = np.empty(len(feats), dtype=int)
    ranks[np.argsort(importance)[::-1]] = np.arange(1, len(feats) + 1)
    info_feats = [
        {"name": f, "min": r6(X_train_df[f].min()), "max": r6(X_train_df[f].max()),
         "mean": r6(X_train_df[f].mean()), "std": r6(X_train_df[f].std()),
         "importance": r6(importance[i]), "rank": int(ranks[i])}
        for i, f in enumerate(feats)
    ]
    proba = models["xgb_baseline"].predict_proba(X_test)[:, 1]
    samples = []
    for label, yv in (("signal", 1), ("background", 0)):
        idx = np.where(y_test == yv)[0]
        conf = proba[idx] if yv == 1 else 1 - proba[idx]
        best = idx[np.argsort(conf)[::-1][:3]]  # most confidently classified examples
        for j, i in enumerate(best, 1):
            samples.append({"label": f"{label} example {j}", "y": int(yv), "values": [r6(v) for v in X_test[i]]})
    return {
        "model_id": model_id, "synthetic": bool(synthetic), "version": "1.0",
        "classes": CLASSES, "n_train": int(n_train), "n_test": int(n_test),
        "features": info_feats, "samples": samples,
    }


def save_figures(results, out_dir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping PNG figures (pip install matplotlib)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = {"xgb_baseline": "XGBoost baseline", "xgb_robust": "XGBoost dropout-augmented",
              "xgb_baseline_mean_impute": "Baseline + mean-impute (knows dead sensors)", "mlp": "MLP"}

    d = results["dropout"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = [r * 100 for r in d["rates"]]
    for name, s in d["series"].items():
        ax.errorbar(x, [v * 100 for v in s["mean"]], yerr=[v * 100 for v in s["std"]], marker="o",
                    capsize=3, label=labels.get(name, name))
    ax.set_xlabel("Sensors set to 0 (%)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy under sensor dropout")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "dropout.png", dpi=160)
    plt.close(fig)

    n = results["noise"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, g in zip(axes, ("top", "bottom")):
        for name, s in n["groups"][g]["series"].items():
            ax.errorbar(n["alphas"], [v * 100 for v in s["mean"]], yerr=[v * 100 for v in s["std"]], marker="o",
                        capsize=3, label=labels.get(name, name))
        ax.set_title(f"Noise on the {n['k']} {'most' if g == 'top' else 'least'} important features")
        ax.set_xlabel("Noise sigma (multiples of feature std)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "noise.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["openml", "uci", "csv", "synthetic"], default="openml")
    p.add_argument("--csv", help="path to a CSV (with --source csv)")
    p.add_argument("--target", default="signal", help="target column name (with --source csv)")
    p.add_argument("--quick", action="store_true", help="small models and few seeds, for smoke tests")
    p.add_argument("--no-mlp", action="store_true", help="skip the MLP baseline")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--artifacts-dir", help="write ALL outputs here instead of the repo locations (used by CI)")
    a = p.parse_args(argv)

    if a.artifacts_dir:
        base = Path(a.artifacts_dir)
        bundle_path = base / "models" / "model_bundle.joblib"
        payload_path, sample_csv = base / "test_payload.json", base / "sample_inputs.csv"
        public_dir, docs_dir = base / "public", base / "docs"
    else:
        bundle_path = ROOT / "backend" / "models" / "model_bundle.joblib"
        payload_path, sample_csv = ROOT / "backend" / "test_payload.json", ROOT / "data" / "sample_inputs.csv"
        public_dir, docs_dir = ROOT / "frontend" / "public", ROOT / "docs"

    print(f"Loading data ({a.source}) ...")
    if a.source == "openml":
        X_df, y = load_openml()
    elif a.source == "uci":
        X_df, y = load_uci(a.csv)
    elif a.source == "csv":
        X_df, y = load_csv(a.csv, a.target)
    else:
        X_df, y = load_synthetic()
    synthetic = a.source == "synthetic"
    print(f"  {len(X_df)} rows, {X_df.shape[1]} features, {y.mean():.1%} signal")

    X_tr_df, X_te_df, y_tr, y_te = train_test_split(X_df, y, test_size=a.test_size, random_state=42, stratify=y)
    X_tr, X_te = X_tr_df.to_numpy(float), X_te_df.to_numpy(float)

    print("Training models ...")
    models = fit_models(X_tr, y_tr, quick=a.quick, with_mlp=not a.no_mlp)

    print("Replication table (clean test set) ...")
    table = rb.replication_table(models, X_te, y_te)
    for r in table:
        print(f"  {r['model']:<14} accuracy {r['accuracy']:.4f}   AUC {r['auc']:.4f}")

    importance = rb.feature_importance(models["xgb_baseline"])
    mean, std = X_tr.mean(axis=0), X_tr.std(axis=0)
    seeds = 2 if a.quick else 5
    rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    alphas = [0, 0.5, 1, 2] if a.quick else [0, 0.25, 0.5, 1, 2, 4]

    print("Dropout experiment ...")
    dropout = rb.dropout_experiment(models, X_te, y_te, mean, rates, seeds)
    print("Noise experiment ...")
    noise = rb.noise_experiment(models, X_te, y_te, std, importance, alphas, seeds)
    for g, gd in noise["groups"].items():
        gd["collapse_alpha"] = {n: rb.collapse_threshold(alphas, s["mean"]) for n, s in gd["series"].items()}

    model_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    features = list(X_df.columns)
    results = {
        "model_id": model_id, "synthetic": synthetic, "source": a.source,
        "n_rows": int(len(X_df)), "n_features": int(X_df.shape[1]),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "split": f"{int((1 - a.test_size) * 100)}/{int(a.test_size * 100)} stratified, random_state=42",
        "replication": table, "dropout": dropout, "noise": noise,
        "top_features": [features[i] for i in noise["groups"]["top"]["columns"]],
        "bottom_features": [features[i] for i in noise["groups"]["bottom"]["columns"]],
    }
    info = build_model_info(X_tr_df, importance, X_te, y_te, models, model_id, synthetic, len(X_tr), len(X_te))

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(build_bundle(models, features, model_id, synthetic), bundle_path, compress=3)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps({f: info["samples"][0]["values"][i] for i, f in enumerate(features)}, indent=2))
    sample_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([s["values"] for s in info["samples"]], columns=features).to_csv(sample_csv, index=False)
    for d in (public_dir, docs_dir):
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text(json.dumps(results, indent=1))
    (public_dir / "model_info.json").write_text(json.dumps(info, indent=1))
    save_figures(results, docs_dir / "figures")

    print(f"\nSaved bundle:  {bundle_path}  ({bundle_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved results: {public_dir / 'results.json'}")
    print("Done." + ("  (synthetic data: results are NOT MiniBooNE results)" if synthetic else ""))


if __name__ == "__main__":
    main()
