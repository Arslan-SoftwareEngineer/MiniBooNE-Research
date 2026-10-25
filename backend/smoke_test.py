"""Manual stage 7.4: call the RUNNING local API with Requests (start Flask first)."""
import json
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent
payload = json.loads((BASE / "test_payload.json").read_text(encoding="utf-8"))

with requests.Session() as session:
    session.trust_env = False  # ignore campus proxy settings for localhost
    health = session.get("http://127.0.0.1:5000/health", timeout=30)
    print("GET /health ->", health.status_code, health.text.strip())
    resp = session.post("http://127.0.0.1:5000/compare", json={"features": payload, "offline": list(payload)[:10]}, timeout=60)
    print("POST /compare ->", resp.status_code)
    print(json.dumps(resp.json(), indent=2)[:1200])
    resp.raise_for_status()
print("PASS: API request")
