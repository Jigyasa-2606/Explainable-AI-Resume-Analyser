# Deploy on Vercel + Render

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
Start Command: uvicorn backend_api:app --host 0.0.0.0 --port $PORT
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
- `trained_match_model.joblib` is committed with the repo so Render does not need to retrain on every deploy.
