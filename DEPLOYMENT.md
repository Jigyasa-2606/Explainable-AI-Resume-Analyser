# Deploy locally with Docker, or on Vercel + Render

## Docker (local)

This is the easiest way to run the current ranking stack (Random Forest API + React UI) on your machine.

### 1. Install Docker Desktop

Download [Docker Desktop](https://www.docker.com/products/docker-desktop/) and confirm it is running:

```bash
docker --version
docker compose version
```

### 2. Put API keys in `.env`

In the project root:

```bash
cp env.example .env
```

Fill in the keys you have. Remotive works without keys. Leave `STAGE2_RERANKER=off`.

### 3. Build and start

From the project root (`Resume analyzer final`):

```bash
docker compose up --build
```

First build can take several minutes (Python packages + the 31 MB model).

### 4. Open the app

- App: [http://localhost:8080](http://localhost:8080)
- API health: [http://localhost:8001/health](http://localhost:8001/health)

Upload a resume and search. The UI talks to nginx on 8080, which proxies `/api` to the FastAPI container.

### Useful commands

```bash
docker compose logs -f backend
docker compose down
docker compose up --build --force-recreate
```

### What the containers are

| Service | Container | Port |
|---------|-----------|------|
| Backend | FastAPI + Random Forest | 8000 |
| Frontend | nginx serving the Vite build | 8080 |

Do not put `sentence-transformers` in the Docker image for this setup. Ranking uses Random Forest only.

---

## Vercel + Render


Use Vercel for the React frontend and Render for the FastAPI backend.

## Quick deploy (Blueprint)

1. Push this repository to GitHub.
2. In Render, choose **New +** → **Blueprint** and connect the repo. `render.yaml` creates the API service.
3. In Vercel, import the same repo with **Root Directory** set to `frontend`.
4. Set `VITE_API_BASE` on Vercel to your Render URL (no trailing slash).
5. Redeploy Vercel after the env var is saved.

## 1. Render Backend

Create a Render **Web Service** from the GitHub repository.

Settings:

```text
Root Directory: .
Runtime: Python
Build Command: pip install -r requirements.txt
Start Command: uvicorn backend.backend_api:app --host 0.0.0.0 --port $PORT
```

Environment variables:

```text
RAPIDAPI_KEY=...
ADZUNA_APP_ID=...
ADZUNA_APP_KEY=...
SERPAPI_KEY=...
JOOBLE_API_KEY=...
FRONTEND_ORIGINS=https://your-vercel-app.vercel.app
```

After deploy, test:

```text
https://your-render-backend.onrender.com/health
```

Expected response:

```json
{"status":"ok"}
```

## 2. Vercel Frontend

Create a Vercel project from the same GitHub repository.

Settings:

```text
Framework Preset: Vite
Root Directory: frontend
Build Command: npm run build
Output Directory: dist
```

Environment variable:

```text
VITE_API_BASE=https://your-render-backend.onrender.com
```

After adding or changing `VITE_API_BASE`, redeploy the Vercel project.

## 3. Connect Both URLs

1. Copy the Render backend URL into Vercel as `VITE_API_BASE`.
2. Copy the Vercel frontend URL into Render as `FRONTEND_ORIGINS`.
3. Redeploy both services.

## 4. Final Test

In the deployed frontend:

1. Upload a PDF, DOCX, or TXT resume.
2. Select `All Providers`.
3. Search jobs.
4. Confirm jobs are fetched, deduplicated, and ranked.
5. If one API provider fails, confirm the app shows provider warnings but still displays jobs from working providers.

## Notes

- Do not deploy `.env`; add secrets in Render environment variables.
- Do not commit `.venv/`, `frontend/node_modules/`, or `frontend/dist/`.
- `backend/trained_match_model.joblib` is committed with the repo so Render does not need to retrain on every deploy.
