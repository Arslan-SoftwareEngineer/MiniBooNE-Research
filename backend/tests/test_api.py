"""API tests. The first four are the ones required by the lab manual (stage 7.6)."""


def test_health(client, features):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["features"] == len(features) and body["model_loaded"] is True


def test_predict_valid(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["prediction"] in (0, 1)
    assert abs(sum(body["probabilities"]) - 1) < 1e-6


def test_predict_missing_feature(client, payload, features):
    incomplete = dict(payload)
    incomplete.pop(features[0])
    assert client.post("/predict", json=incomplete).status_code == 400


def test_predict_invalid_json(client):
    r = client.post("/predict", data="not-json", content_type="application/json")
    assert r.status_code == 400


# ---- project-specific tests ------------------------------------------------
def test_predict_both_models(client, payload):
    for key in ("baseline", "robust"):
        r = client.post(f"/predict?model={key}", json=payload)
        assert r.status_code == 200 and r.get_json()["model"] == key


def test_predict_unknown_model(client, payload):
    assert client.post("/predict?model=nope", json=payload).status_code == 400


def test_predict_rejects_non_numeric(client, payload, features):
    bad = dict(payload)
    bad[features[1]] = "abc"
    r = client.post("/predict", json=bad)
    assert r.status_code == 400 and features[1] in r.get_json()["invalid"]


def test_predict_rejects_extra_feature(client, payload):
    r = client.post("/predict", json={**payload, "bogus": 1})
    assert r.status_code == 400 and r.get_json()["extra"] == ["bogus"]


def test_compare_structure(client, payload, features):
    offline = features[:5]
    r = client.post("/compare", json={"features": payload, "offline": offline})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["n_offline"] == 5
    for key in ("baseline", "robust"):
        for state in ("intact", "degraded"):
            p = body["models"][key][state]["probability_signal"]
            assert 0.0 <= p <= 1.0


def test_compare_no_offline_means_identical_readings(client, payload):
    body = client.post("/compare", json={"features": payload, "offline": []}).get_json()
    for key in ("baseline", "robust"):
        m = body["models"][key]
        assert m["intact"]["probability_signal"] == m["degraded"]["probability_signal"]


def test_compare_unknown_sensor(client, payload):
    r = client.post("/compare", json={"features": payload, "offline": ["not_a_sensor"]})
    assert r.status_code == 400


def test_no_model_returns_503(tmp_path):
    from app import create_app

    c = create_app(tmp_path / "missing.joblib").test_client()
    assert c.get("/health").get_json()["model_loaded"] is False
    assert c.post("/predict", json={}).status_code == 503
