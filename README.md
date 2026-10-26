# Neutrino classifier under sensor failure (MiniBooNE)

A full-stack research prototype: **ML model → Flask REST API → Angular app → GitHub → CI/CD → GitHub Pages**.

The research question: *when some of the 50 detector sensors die, how much accuracy does a normally trained XGBoost model lose, and does training on dropout-augmented data help?* The app lets you switch sensors off one by one (or in groups) and compare the **baseline** and **dropout-trained** models side by side, and a **Study results** tab shows the measured comparison over the whole test set.

```
Angular (localhost:4200) --/api/compare--> dev proxy --> Flask (127.0.0.1:5000) --> model_bundle.joblib
GitHub push --> Actions CI (pytest + training smoke test + Angular build) --> Pages deploy (static Angular)
```

## What is where

| Path | Purpose |
|---|---|
| `backend/train.py` | Loads MiniBooNE, trains baseline XGBoost, dropout-trained XGBoost and an MLP, runs the robustness experiments, writes every artifact |
| `backend/robustness.py` | Corruption functions (dropout, noise, augmentation) and the experiments |
| `backend/app.py` | Flask API: `GET /health`, `POST /predict`, `POST /compare` |
| `backend/tests/` | pytest suite (API, corruptions, repo structure) |
| `frontend/` | Angular app (sensor grid, side-by-side comparison, study charts) |
| `.github/workflows/ci.yml` | Continuous integration on every push / pull request |
| `.github/workflows/pages.yml` | Continuous deployment to GitHub Pages after CI passes |
| `render.yaml` | Optional: hosts the Flask API so the public site can score readings |
| `docs/` | Results JSON, figures for the report, evidence screenshots |

## Run it locally (Windows, VS Code, PowerShell)

Prerequisites: Python 3.12, Node.js 24 LTS (Angular CLI 22 needs Node 22.22.3+ or 24.15+), Git.

**1. Python environment** (from the project root, terminal 3)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
python -m pip install --upgrade pip
python -m pip install -r backend\requirements-train.txt
```

**2. Put the dataset in `data\MiniBooNE_PID.csv`, then train** (a few minutes)

```powershell
python backend\train.py
```

`train.py` reads `data\MiniBooNE_PID.csv` and detects its layout on its own: the original UCI layout (first line = two counts), a CSV with a header and a label column, or a headerless CSV with the label last. If it cannot find the label column, name it: `python backend\train.py --target COLUMN_NAME`. Without a local file it downloads MiniBooNE from OpenML, then from UCI. Add `--no-mlp` to skip the MLP.
After training it prints the xgboost / scikit-learn versions: put those in `backend/requirements.txt` so CI and the hosted API can load the saved model.
This overwrites the demo files that ship in the repo (model bundle, `frontend/public/model_info.json`, `results.json`, sample payload, figures in `docs/figures/`).

**3. Run the tests**

```powershell
python -m pytest -q backend\tests
```

**4. Terminal 1: Flask** (keep it open)

```powershell
.\.venv\Scripts\Activate.ps1
python backend\app.py
```
Check `http://127.0.0.1:5000/health` shows `"model_loaded": true`.

**5. Terminal 2: Angular** (keep it open)

```powershell
cd frontend
npm install
npm start
```
Open `http://localhost:4200`. The dev proxy (`frontend/proxy.conf.json`, already wired into `angular.json`) forwards `/api/*` to Flask, so no CORS setup is needed.
After retraining, restart Flask and reload the page (the form reads `model_info.json`; the app warns if it no longer matches the API's model).

Optional check of the running API: `python backend\smoke_test.py`.

## Put it on GitHub with CI/CD

```powershell
git init
git branch -M main
git add .
git status --short          # make sure .venv and node_modules are NOT listed
git commit -m "Initial full-stack ML application"
git remote add origin https://github.com/Arslan-SoftwareEngineer/YOUR_REPO.git
git push -u origin main
```

Create the GitHub repository first (public, **without** README / .gitignore / license). Then:

1. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
2. Open **Actions**. `Full-stack CI` runs on every push; when it succeeds, `Deploy Angular to GitHub Pages` runs automatically. Site: `https://Arslan-SoftwareEngineer.github.io/YOUR_REPO/`.
3. From now on every `git push` to `main` is tested and, if green, deployed. A failing test blocks the deployment.

### Optional: live predictions on the public site

GitHub Pages cannot run Python, so by default the public site shows the study results and a preview-only sensor grid. To enable live scoring, host the API too:

1. Commit the trained bundle (`backend/models/model_bundle.joblib`; it is not git-ignored).
2. On [render.com](https://render.com) create a *Blueprint* from this repo (it reads `render.yaml`). Set `CORS_ORIGINS` to `https://Arslan-SoftwareEngineer.github.io`. Render redeploys on every push.
3. In GitHub: **Settings → Secrets and variables → Actions → Variables → New repository variable** `API_URL` = your Render URL (no trailing slash). Re-run the Pages workflow. (Free instances sleep, so the first request can take about a minute.)

## How the research maps to the code

- **Replication**: `train.py` uses a stratified 80/20 split (`random_state=42`), XGBoost (400 trees, depth 6) and a scaled MLP (128-64). To match your own tuned notebook values, edit `make_xgb` / `make_mlp` in `backend/robustness.py`. Compare against the table in the paper you replicate and report any protocol differences.
- **Sensor dropout**: a random subset (10–40%) of sensors reads `0` for the whole test set; 5 random subsets per rate, mean ± sd.
- **Targeted noise**: Gaussian noise (σ = multiples of each feature's training std) on the 5 most vs 5 least important features (XGBoost gain, ranked on the training set).
- **Mitigation**: the robust model trains on clean data plus a corrupted copy in which every row loses its own random 10–40% of sensors.
- **Extra reference**: baseline + mean-fill, which assumes the system knows which sensors died.

Caveats to state in the report: dropout curves favour the dropout-trained model by construction (it trains on that failure type), so the noise experiment is the fairer generalisation test; "zero" is one failure mode; only accuracy is reported.

## Lab manual stage map

| Manual stage | Where |
|---|---|
| 1–3 plan, tools, workspace | this README, `docs/environment.txt`, `.venv` |
| 4–5 model bundle, payload | `train.py` writes `model_bundle.joblib`, `backend/test_payload.json`, `data/sample_inputs.csv` |
| 6–7 Flask + tests | `backend/app.py`, `backend/tests/`, `backend/smoke_test.py` |
| 8–10 Angular + integration | `frontend/` (proxy already configured) |
| 11–12 Git + GitHub | commands above |
| 13 CI | `.github/workflows/ci.yml` |
| 14 Pages | `.github/workflows/pages.yml` |
| 15 submission | repo URL, Actions run URLs, Pages URL, screenshots in `docs/`, this README, report |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `train.py` fails inside `sklearn\datasets\_openml.py` (for example `got 'bool'`) | Update with the latest `backend/train.py` (it retries with a tolerant parser, then falls back to the UCI file). Delete `data\openml_cache` and run again. |
| `getaddrinfo failed` / `URLError` | Your machine cannot resolve the host (internet, VPN or campus proxy). Not an issue when `data\MiniBooNE_PID.csv` exists: it is used first and needs no network. |
| You already have the data as a CSV | `python backend\train.py --source csv --csv PATH --target LABEL_COLUMN` (label values `True`/`1`/`signal` count as signal). |
| Angular shows "Flask is running but has no trained model" | Run `python backend\train.py`, restart Flask, press "Check again". |
| Port 5000 already in use | Stop the old Flask process, or change the port in `app.py` and `frontend/proxy.conf.json`. |

Research/teaching prototype only; not a validated physics analysis.
