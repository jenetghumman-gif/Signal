# PR Intelligence Engine

A full-stack AI-powered PR analysis tool. Paste any news article or brand mention and get sentiment scoring, reputation risk assessment, and a drafted comms response — powered by Claude.

## Project structure

```
pr-intelligence/
├── backend/
│   ├── main.py            # FastAPI app
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    └── index.html         # Single-file frontend
```

## Setup

### 1. Backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Add your API key
cp .env.example .env
# Open .env and replace with your real Anthropic API key

# Run the server
uvicorn main:app --reload
# Server runs at http://localhost:8000
```

### 2. Frontend

Open `frontend/index.html` directly in your browser — no build step needed.

Make sure the `API_BASE` at the top of the script tag matches where your backend is running (default: `http://localhost:8000`).

## API

### POST /analyze

**Request body:**
```json
{
  "text": "Article or mention text here...",
  "industry": "a technology company"
}
```

**Response:**
```json
{
  "sentiment": "Negative",
  "sentimentScore": -72,
  "reputationRisk": 9,
  "urgency": "Critical",
  "urgencyReason": "Safety concerns and ongoing media coverage require immediate response.",
  "signals": [
    { "label": "Safety recall risk", "type": "negative" },
    { "label": "Whistleblower leak", "type": "negative" },
    { "label": "Stock price drop", "type": "negative" }
  ],
  "summary": "...",
  "draftResponse": "..."
}
```

### GET /health

Returns `{ "status": "ok" }` — useful for deployment health checks.

## Deployment

For production, deploy the FastAPI backend to **Azure App Service** or **AWS Lambda** (via Mangum). Set `ANTHROPIC_API_KEY` as an environment variable in your cloud provider's config — never commit it to source control.

Update `API_BASE` in `index.html` to your deployed backend URL, then host the frontend on any static host (Azure Static Web Apps, S3, Netlify, etc.).
