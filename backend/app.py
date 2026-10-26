"""Flask REST API for the MiniBooNE baseline-vs-robust demo.

Endpoints
    GET  /health                     status, number of features, model id
    POST /predict[?model=baseline]   flat JSON {feature: value, ...} -> one prediction
    POST /compare                    {"features": {...}, "offline": [names]} -> baseline vs robust,
                                     on the intact reading and on the degraded reading
                                     (offline sensors read 0)

Configuration (environment variables)
    MODEL_BUNDLE   path to the .joblib bundle (default backend/models/model_bundle.joblib)
    CORS_ORIGINS   comma-separated allowed browser origins, e.g. https://USER.github.io
                   (not needed locally: the Angular dev proxy avoids CORS)
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

BASE = Path(__file__).resolve().parent
DEFAULT_BUNDLE = BASE / "models" / "model_bundle.joblib"
DISCLAIMER = "Research/teaching prototype. Not a validated physics analysis."


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def create_app(bundle_path=None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024

    path = Path(bundle_path or os.environ.get("MODEL_BUNDLE") or DEFAULT_BUNDLE)
    bundle = joblib.load(path) if path.exists() else None  # trusted file created by us
    features = bundle["features"] if bundle else []
    models = bundle["models"] if bundle else {}
    classes = bundle.get("classes", {}) if bundle else {}

    allowed = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]

    @app.before_request
    def _preflight():
        if request.method == "OPTIONS":
            return app.make_response(("", 204))

    @app.after_request
    def _cors(resp):
        origin = request.headers.get("Origin")
        if origin and ("*" in allowed or origin in allowed):
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            resp.headers["Vary"] = "Origin"
        return resp

    def no_model():
        return jsonify(error="No model loaded. Run: python backend/train.py"), 503

    def parse_features(data):
        """Return (row, error_response). Row is a 1 x n float array in model order."""
        if not isinstance(data, dict):
            return None, (jsonify(error="Expected a JSON object"), 400)
        missing = [f for f in features if f not in data]
        extra = [f for f in data if f not in features]
        if missing or extra:
            return None, (jsonify(error="Input schema mismatch", missing=missing, extra=extra), 400)
        bad = [f for f in features if not _is_number(data[f])]
        if bad:
            return None, (jsonify(error="Values must be finite numbers", invalid=bad), 400)
        return np.array([[float(data[f]) for f in features]]), None

    def run(model_key, row):
        proba = models[model_key].predict_proba(row)[0]
        p_signal = float(proba[1])
        label = int(p_signal >= 0.5)
        return {"prediction": label, "label": classes.get(str(label), str(label)),
                "probability_signal": p_signal, "probabilities": [float(proba[0]), p_signal]}

    @app.get("/health")
    def health():
        return jsonify(status="ok", model_loaded=bundle is not None, features=len(features),
                       task=bundle.get("task") if bundle else None, models=list(models),
                       model_id=bundle.get("model_id") if bundle else None,
                       synthetic=bool(bundle.get("synthetic")) if bundle else None)

    @app.post("/predict")
    def predict():
        if bundle is None:
            return no_model()
        key = request.args.get("model", "baseline")
        if key not in models:
            return jsonify(error="Unknown model", allowed=list(models)), 400
        row, err = parse_features(request.get_json(silent=True))
        if err:
            return err
        try:
            result = run(key, row)
        except (ValueError, TypeError) as exc:
            return jsonify(error="Invalid input", detail=str(exc)), 400
        except Exception:
            app.logger.exception("Prediction failed")
            return jsonify(error="Prediction processing failed"), 500
        return jsonify(**result, model=key, task=bundle["task"], disclaimer=DISCLAIMER)

    @app.post("/compare")
    def compare():
        if bundle is None:
            return no_model()
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or "features" not in body:
            return jsonify(error='Expected {"features": {...}, "offline": [...]}'), 400
        offline = body.get("offline", [])
        if not isinstance(offline, list) or any(o not in features for o in offline):
            unknown = [o for o in offline if o not in features] if isinstance(offline, list) else offline
            return jsonify(error="Unknown sensor names in 'offline'", unknown=unknown), 400
        row, err = parse_features(body["features"])
        if err:
            return err
        degraded = row.copy()
        for name in offline:
            degraded[0, features.index(name)] = 0.0  # dead sensor reads 0, as in training
        try:
            out = {k: {"intact": run(k, row), "degraded": run(k, degraded)} for k in models}
        except Exception:
            app.logger.exception("Comparison failed")
            return jsonify(error="Prediction processing failed"), 500
        return jsonify(models=out, offline=sorted(set(offline)), n_offline=len(set(offline)),
                       model_id=bundle.get("model_id"), disclaimer=DISCLAIMER)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
