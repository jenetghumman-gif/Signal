import os
import re
import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="PR Intelligence Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend domain in production
    allow_methods=["POST"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


class AnalyzeRequest(BaseModel):
    text: str
    industry: str = "a brand"


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    prompt = f"""You are a senior PR strategist. Analyze the following media mention about "{req.industry}".

Return ONLY a valid JSON object, no markdown, no preamble:
{{
  "sentiment": "Positive" or "Neutral" or "Negative",
  "sentimentScore": number -100 to 100,
  "reputationRisk": number 0 to 10,
  "urgency": "Low" or "Medium" or "High" or "Critical",
  "urgencyReason": "one short sentence",
  "signals": [{{"label": "max 3 words", "type": "positive" or "negative" or "neutral"}}],
  "summary": "2-3 sentence factual summary and brand implications",
  "draftResponse": "Professional 3-5 sentence spokesperson statement in first person. Tone matches urgency."
}}

Article:
{req.text}"""

    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}],
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(ANTHROPIC_URL, headers=headers, json=payload)

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    raw = "".join(block.get("text", "") for block in data.get("content", []))

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise HTTPException(status_code=500, detail="No JSON found in model response")

    try:
        result = json.loads(match.group())
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON parse error: {e}")

    return result


@app.get("/health")
async def health():
    return {"status": "ok"}
