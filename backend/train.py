"""Train baseline + dropout-robust models on MiniBooNE and run the robustness study.

Examples (run from the project root, venv active):

    python backend/train.py                      # uses data/MiniBooNE_PID.csv if present, else OpenML/UCI
    python backend/train.py --target signal      # name the label column if it is not auto-detected
    python backend/train.py --source uci         # original UCI file instead
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
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/199/miniboone+particle+identification.zip"
OPENML_ID = 41150  # "MiniBooNE" on OpenML: 130,064 rows, 50 features, target = signal
DEFAULT_DATA = ROOT / "data" / "MiniBooNE_PID.csv"  # put your dataset here
N_TOTAL, N_SIGNAL = 130_064, 36_499  # known MiniBooNE sizes (UCI file: signal rows first)
CLASSES = {"0": "background (muon neutrino)", "1": "signal (electron neutrino)"}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_openml():
    """OpenML copy of MiniBooNE. Tries sklearn's fast pandas parser, then the slower but
    more tolerant liac-arff parser (the pandas one crashes on some pandas/sklearn combinations)."""
    from sklearn.datasets import fetch_openml

    home = str(ROOT / "data" / "openml_cache")
    last = None
    for parser in ("auto", "liac-arff"):
        try:
            ds = fetch_openml(data_id=OPENML_ID, as_frame=True, parser=parser, data_home=home)
            break
        except Exception as exc:  # noqa: BLE001 - we report and fall through
            last = exc
            print(f"  OpenML parser '{parser}' failed: {type(exc).__name__}: {str(exc)[:150]}")
    else:
        raise RuntimeError("OpenML download/parsing failed") from last
    X = ds.data.astype(float)
    y = ds.target.astype(str).str.lower().isin(["true", "1", "signal"]).astype(int).to_numpy()
    return X, y


def load_auto(target=None, signal_rows=None):
    """Local file data/MiniBooNE_PID.csv if present, else OpenML, else the UCI download."""
    if DEFAULT_DATA.exists():
        print(f"  using local file {DEFAULT_DATA}")
        return load_file(DEFAULT_DATA, target, signal_rows)
    try:
        return load_openml()
    except Exception as exc:  # noqa: BLE001
        print(f"\nOpenML failed ({type(exc).__name__}). Falling back to the UCI file ...\n")
        try:
            return load_uci()
        except Exception as exc2:  # noqa: BLE001
            raise SystemExit(
                "Could not fetch MiniBooNE automatically "
                f"({type(exc2).__name__}: {exc2}).\n"
                "Check your internet/DNS/VPN/proxy, or provide the data yourself:\n"
                "  * UCI file: download MiniBooNE_PID.txt, save it as data\\MiniBooNE_PID.txt, then run\n"
                "      python backend\\train.py --source uci\n"
                "  * your own file: save it as data\\MiniBooNE_PID.csv and run python backend\\train.py"
            ) from exc2


def load_uci(path: str | None = None):
    import urllib.request

    target = Path(path) if path else ROOT / "data" / "MiniBooNE_PID.txt"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            print(f"Downloading {UCI_URL} ...")
            urllib.request.urlretrieve(UCI_URL, target)
        except Exception as exc:  # noqa: BLE001
            print(f"  direct download failed ({type(exc).__name__}); trying the UCI zip ...")
            import io
            import zipfile

            with urllib.request.urlopen(UCI_ZIP_URL, timeout=120) as resp:
                zf = zipfile.ZipFile(io.BytesIO(resp.read()))
            name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
            target.write_bytes(zf.read(name))
    with open(target, encoding="utf-8-sig", errors="replace") as f:
        counts = f.readline().replace(",", " ").split()
        second = f.readline()
    n_sig, n_bkg = (int(float(v)) for v in counts[:2])
    sep = "," if "," in second else r"\s+"
    df = pd.read_csv(target, sep=sep, skiprows=1, header=None, encoding="utf-8-sig")
    df = df.dropna(axis=1, how="all").iloc[: n_sig + n_bkg]
    y = np.r_[np.ones(n_sig, dtype=int), np.zeros(n_bkg, dtype=int)]
    X = df.astype(float)
    X.columns = [f"ParticleID_{i}" for i in range(X.shape[1])]
    bad = (X == -999).any(axis=1).to_numpy()
    if bad.any():
        print(f"Dropping {bad.sum()} rows containing the -999 'missing' code")
        X, y = X[~bad].reset_index(drop=True), y[~bad]
    return X, y


def _isint(s: str) -> bool:
    try:
        v = float(s)
        return v == int(v)
    except (ValueError, OverflowError):
        return False


def _isnum(s: str) -> bool:
    try:
        float(s.strip())
        return True
    except ValueError:
        return False


def to_binary(col: pd.Series) -> np.ndarray:
    """Turn a two-class label column into 1 = signal, 0 = background."""
    if col.dtype == bool:
        return col.astype(int).to_numpy()
    vals = pd.unique(col.dropna())
    if pd.api.types.is_numeric_dtype(col):
        s = set(float(v) for v in vals)
        if s <= {0.0, 1.0}:
            return col.astype(int).to_numpy()
        if s <= {-1.0, 1.0}:
            return (col == 1).astype(int).to_numpy()
        raise SystemExit(f"Label column has numeric values {sorted(s)[:6]}; expected 0/1 (1 = signal).")
    low = col.astype(str).str.strip().str.lower()
    sig = {"true", "1", "1.0", "signal", "s", "yes", "electron", "nue"}
    if not set(low.unique()) & sig:
        raise SystemExit(f"Cannot tell which label means signal. Values found: {list(vals)[:6]}. "
                         "Rename the signal value to 'signal' or 1.")
    return low.isin(sig).astype(int).to_numpy()


def _finish(X: pd.DataFrame, y: np.ndarray):
    """Shared tail: drop rows with the -999 'missing' code."""
    bad = (X == -999).any(axis=1).to_numpy()
    if bad.any():
        print(f"  dropping {bad.sum()} rows containing the -999 'missing' code")
        X, y = X[~bad].reset_index(drop=True), y[~bad]
    return X, y


def check_positional_labels(X: pd.DataFrame, y: np.ndarray):
    """Labels built from row ORDER are only valid if the file is still in its original order.
    A quick model must beat the majority class clearly, otherwise the order was lost (shuffled/sorted)."""
    Xa = X.to_numpy(float)
    Xtr, Xte, ytr, yte = train_test_split(Xa, y, test_size=0.2, random_state=0, stratify=y)
    acc = float(rb.make_xgb(quick=True).fit(Xtr, ytr).score(Xte, yte))
    major = float(max(y.mean(), 1 - y.mean()))
    print(f"  sanity check of row-order labels: quick model accuracy {acc:.3f} (always-background would score {major:.3f})")
    if acc < major + 0.05:
        raise SystemExit(
            "The file has no label column and the row order does not look like the original MiniBooNE order "
            "(signal rows first), so the labels cannot be trusted.\n"
            "Use the original UCI file (MiniBooNE_PID.txt), or a CSV that contains a label column, "
            "or tell me how many leading rows are signal with --signal-rows N if the order is signal-first.")


def load_file(path, target: str | None = None, signal_rows: int | None = None):
    """Read MiniBooNE from a local file and detect its layout automatically.

    Supported: (a) the original UCI layout (first line = '<n_signal> <n_background>', whitespace
    separated, signal rows first), (b) CSV/whitespace table with a header and a label column,
    (c) the same without a header (label = last column), (d) a table of 50 feature columns with NO
    label (for example saved with pandas): labels come from the known UCI row order, signal first.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}\nPut your file at data\\MiniBooNE_PID.csv (or pass --csv PATH).")
    with open(path, "r", errors="replace", encoding="utf-8-sig") as f:  # utf-8-sig strips an invisible BOM
        first = f.readline().strip()
        second = f.readline().strip()
    toks = first.replace(",", " ").split()
    if len(toks) == 2 and all(_isint(x) for x in toks) and len(second.replace(",", " ").split()) >= 20:
        print("  layout: original UCI file (count line + whitespace-separated rows)")
        return load_uci(str(path))

    sep = "," if "," in first else r"\s+"
    cells = [c.strip() for c in (first.split(",") if sep == "," else first.split())]
    all_numeric = all(_isnum(c) for c in cells)
    # pandas writes a header '0,1,2,...,49' when the columns were never named: that is a header row, not data
    index_header = all_numeric and cells == [str(i) for i in range(len(cells))]
    has_header = (not all_numeric) or index_header
    df = pd.read_csv(path, sep=sep, header=0 if has_header else None, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns] if has_header else [f"col_{i}" for i in range(df.shape[1])]
    df = df.loc[:, ~df.columns.str.lower().str.startswith("unnamed")]  # stray index column
    kind = "numbered header row" if index_header else ("header" if has_header else "no header")
    print(f"  layout: {kind}, {df.shape[0]} rows x {df.shape[1]} columns")
    generic_names = index_header or not has_header

    cols = list(df.columns)
    if target:
        if target not in cols:
            raise SystemExit(f"--target '{target}' not found. Columns: {cols[:8]} ... {cols[-3:]}")
    else:
        lower = {c.lower(): c for c in cols}
        target = next((lower[n] for n in ("signal", "target", "label", "class", "y", "is_signal", "pid") if n in lower), None)
        if target is None and df[cols[-1]].nunique(dropna=True) <= 2:
            target = cols[-1]
        if target is None:
            n_sig = signal_rows
            if n_sig is None and df.shape == (N_TOTAL, 50):
                n_sig = N_SIGNAL  # the known MiniBooNE shape: UCI order = first 36,499 rows are signal
            if n_sig is None:
                raise SystemExit(f"Could not find a label column among {df.shape[1]} columns "
                                 f"(first line of the file starts: {first[:60]!r}; "
                                 f"last column has {df[cols[-1]].nunique()} distinct values).\n"
                                 "Pass the label column with --target NAME, or, if the file has only the 50 features "
                                 "in the original order (signal first), pass --signal-rows N.")
            print(f"  no label column: taking the first {n_sig} of {len(df)} rows as signal (original UCI order)")
            y = np.r_[np.ones(n_sig, dtype=int), np.zeros(len(df) - n_sig, dtype=int)]
            X = df.apply(pd.to_numeric, errors="raise").astype(float)
            X.columns = [f"ParticleID_{i}" for i in range(X.shape[1])]
            check_positional_labels(X, y)
            return _finish(X, y)
    print(f"  label column: {target}")
    y = to_binary(df[target])
    X = df.drop(columns=[target]).apply(pd.to_numeric, errors="raise").astype(float)
    if generic_names:
        X.columns = [f"ParticleID_{i}" for i in range(X.shape[1])]
    return _finish(X, y)


def load_csv(path: str, target: str | None = None):  # kept for backwards compatibility
    return load_file(path, target)


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
    p.add_argument("--source", choices=["auto", "file", "csv", "openml", "uci", "synthetic"], default="auto",
                   help="auto = data/MiniBooNE_PID.csv if present, else OpenML, else UCI")
    p.add_argument("--csv", default=str(DEFAULT_DATA), help="dataset path (default data/MiniBooNE_PID.csv)")
    p.add_argument("--target", default=None, help="label column name (auto-detected if omitted)")
    p.add_argument("--signal-rows", type=int, default=None,
                   help="for a file with NO label column: the first N rows are signal (default: 36499 for the full dataset)")
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
    if a.source == "auto":
        X_df, y = load_auto(a.target, a.signal_rows)
    elif a.source == "openml":
        X_df, y = load_openml()
    elif a.source == "uci":
        X_df, y = load_uci(None if a.csv == str(DEFAULT_DATA) else a.csv)
    elif a.source in ("file", "csv"):
        X_df, y = load_file(a.csv, a.target, a.signal_rows)
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
    import flask  # noqa: F401  (only to report its version)
    import sklearn
    import xgboost

    print("\nIf you deploy (CI / Render), pin these versions in backend/requirements.txt so the bundle loads there:")
    print(f"  xgboost=={xgboost.__version__}   scikit-learn=={sklearn.__version__}   numpy=={np.__version__}   pandas=={pd.__version__}")
    print("Done." + ("  (synthetic data: results are NOT MiniBooNE results)" if synthetic else ""))


if __name__ == "__main__":
    main()