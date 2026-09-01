# Explainable AI Resume Analyser

Upload a resume. The app searches live jobs, ranks them against you with a trained Random Forest model, and shows **why** each role fits — skills, ATS, and match score.

**Live app:** [explainable-ai-resume-analyserr.vercel.app](https://explainable-ai-resume-analyserr.vercel.app)

## Screenshots

Home

![Home](docs/screenshots/home.jpg)

Upload a resume and set search preferences

![Search](docs/screenshots/search.jpg)

## How to use

1. Open the live app (or run it locally).
2. Upload a PDF, DOCX, or TXT resume — or paste the text.
3. Optionally pick country, job vs internship, and up to 5 roles to emphasise.
4. Click **Find matching jobs**.
5. Read the ranked cards: fit score, matched / missing skills, ATS notes, and a short explanation.

The first search after a pause can take a minute. The API sleeps on the free Render plan.

## What we built

- Reads PDF, DOCX, or TXT resumes
- Pulls live jobs from JSearch, Adzuna, SerpAPI, Jooble, Internshala (Apify), and Remotive
- Ranks them with a trained Random Forest and explains skills, ATS, and fit
- If one job board fails, results from the others still appear

## How ranking works
Each resume–job pair is turned into features (skill overlap, related skills, ATS signals, section quality, text similarity). A **Random Forest** trained on ~10k labeled pairs predicts the match score. The UI shows those features back as the explanation, not a black-box number.
Ranking quality was measured with **NDCG@5**. A second-stage SBERT reranker was tried and left **off** — on this labeled set, RF already ranked better, so live scoring stays RF-only.

## Tech stack

| Layer | Tech |
|-------|------|
| Frontend | React 18, Vite |
| Backend | Python, FastAPI, Uvicorn |
| ML | scikit-learn Random Forest, joblib |
| Resume parsing | pypdf, PyMuPDF, python-docx |
| Job APIs | JSearch, Adzuna, SerpAPI, Jooble, Apify, Remotive |
| Local run | Docker Compose (nginx + API) |
| Hosting | Vercel (UI), Render (API) |

## Run locally

You need [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
cp env.example .env
```

Put any job-API keys you have in `.env`. Remotive works with no keys. Never commit `.env`.

```bash
docker compose up --build
```

Open [http://localhost:8080](http://localhost:8080)

Stop with `Ctrl+C` or `docker compose down`.

Without Docker:

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

Open [http://localhost:5173](http://localhost:5173)

## Repo layout

```text
backend/     API, ranking, job fetchers, trained model
frontend/    React UI
docs/        README screenshots
```

Public hosting: Vercel (UI) + Render (API). Details are in [DEPLOYMENT.md](DEPLOYMENT.md).
