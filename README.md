# PR Intelligence Engine

A full-stack AI-powered PR analysis tool — no API key required.
Uses HuggingFace `transformers` models running entirely on your machine.

## Models used (auto-downloaded on first run, ~1.5GB total)

| Task | Model |
|------|-------|
| Sentiment analysis | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Urgency + signal detection | `facebook/bart-large-mnli` (zero-shot classification) |
| Summarization | `facebook/bart-large-cnn` |

All models run locally — nothing leaves your machine.

## Quick start

### 1. Backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload
# Runs at http://localhost:8000
# First run will download models (~1.5GB) — this takes a few minutes
```

### 2. Frontend

Just open `frontend/index.html` in your browser. No build step needed.

The `API_BASE` at the top of the script points to `http://localhost:8000` by default.

## API reference

### POST /analyze
```json
{
  "text": "Article text here...",
  "industry": "a technology company"
}
```

Returns:
```json
{
  "sentiment": "Negative",
  "sentimentScore": -82,
  "reputationRisk": 9,
  "urgency": "Critical",
  "urgencyReason": "Immediate public attention demands a rapid, coordinated response.",
  "signals": [
    { "label": "Safety Recall Risk", "type": "negative" }
  ],
  "summary": "...",
  "draftResponse": "..."
}
```

### GET /health
Returns `{ "status": "ok" }` — use for deployment health checks.

## Deployment (portfolio / demo)

1. Deploy the FastAPI backend to **Azure App Service** or **AWS App Runner**
2. Set `API_BASE` in `frontend/index.html` to your deployed URL
3. Host the frontend on **Azure Static Web Apps**, **Netlify**, or **S3 + CloudFront**

No environment variables or secrets needed — the models are baked in.
