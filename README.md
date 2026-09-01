# Explainable AI Resume Analyser

Upload a resume. The app fetches live jobs, ranks them with a trained Random Forest model, and explains every score (skills, ATS, semantic fit, bias).

**Public demo**
- App: [explainable-ai-resume-analyser.vercel.app](https://explainable-ai-resume-analyser.vercel.app)
- API health: `https://resume-analyzer-api-uwle.onrender.com/health`

**Local Docker:** `http://localhost:8080` after `docker compose up --build`

## What it does

- Reads PDF, DOCX, or TXT resumes
- Pulls live jobs from JSearch, Adzuna, SerpAPI, Jooble, Internshala (Apify), and Remotive
- Ranks jobs with a **Random Forest** trained on ~10k resume–job pairs (R² ≈ 0.83, NDCG@5 ≈ 0.99 on the labeled CSV)
- Shows matched / missing skills, ATS hints, and a short explanation per job
- If one job API fails, results from the others still appear

## Tech stack

| Layer | Stack |
|-------|-------|
| Frontend | React 18 + Vite |
| Backend | FastAPI |
| ML | scikit-learn Random Forest (`backend/trained_match_model.joblib`) |
| Local run | Docker Compose |
| Public host | Vercel (UI) + Render (API) |

## Quick start (Docker)

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and start it.
2. Copy keys (never commit `.env`):

```bash
cp env.example .env
```

3. From this folder:

```bash
docker compose up --build
```

4. Open [http://localhost:8080](http://localhost:8080)

API health: [http://localhost:8001/health](http://localhost:8001/health)

Stop with `Ctrl+C` or `docker compose down`.

Remotive works with no keys. Other providers need keys in `.env`.

## Run without Docker

```bash
pip install -r requirements-prod.txt
cp env.example .env
uvicorn backend.backend_api:app --reload
```

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Train / evaluate

```bash
python3 train_match_model.py --dataset resume_job_matching_dataset.csv
python3 evaluate_ranking.py --limit 500 --group-size 20 --k 5,10
```

The evaluator reports **NDCG@5** and **NDCG@10**. On the current labels, Random Forest beats the old formula and beats SBERT/LTR rerank, so those stay off by default (`STAGE2_RERANKER=off`).

## Public deploy (so other people can open it)

Docker on your laptop is not public. To share a URL:

1. Push this repo to GitHub (include `backend/trained_match_model.joblib`).
2. Render: web service from the repo, `render.yaml`, add API keys as env vars.
3. Vercel: import the same repo, **Root Directory** = `frontend`, set `VITE_API_BASE` to the Render URL.
4. Set `FRONTEND_ORIGINS` on Render to your Vercel URL.

Full steps: [`DEPLOYMENT.md`](DEPLOYMENT.md).

## Project layout

```text
backend/                          FastAPI + ranking + job APIs + model
  backend_api.py
  resume_analyzer.py
  job_sources.py
  trained_match_model.joblib
frontend/                          React UI
train_match_model.py               Train Random Forest
evaluate_ranking.py                NDCG@5 / NDCG@10
Dockerfile                         API image
docker-compose.yml                 Local Docker
render.yaml                        Render blueprint
```

## License

Personal / portfolio project.
