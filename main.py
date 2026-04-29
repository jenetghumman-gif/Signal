"""
PR Intelligence Engine — Backend
Uses HuggingFace transformers locally. No API key required.

Models used (auto-downloaded on first run, ~1-2GB total):
  - Sentiment:  cardiffnlp/twitter-roberta-base-sentiment-latest
  - Zero-shot classification (urgency/risk): facebook/bart-large-mnli
  - Summarization: facebook/bart-large-cnn
  - Text generation (draft response): mistralai/Mistral-7B-Instruct-v0.2
    (falls back to a template if GPU/RAM is insufficient)
"""

import re
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="PR Intelligence Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy-load pipelines so the server starts instantly
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_sentiment_pipeline():
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
        max_length=512,
    )

@lru_cache(maxsize=1)
def get_zero_shot_pipeline():
    from transformers import pipeline
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        truncation=True,
    )

@lru_cache(maxsize=1)
def get_summarization_pipeline():
    from transformers import pipeline
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        truncation=True,
    )


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str
    industry: str = "a brand"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SENTIMENT_LABEL_MAP = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral":  "Neutral",
    "LABEL_0":  "Negative",   # fallback for older model label schemes
    "LABEL_1":  "Neutral",
    "LABEL_2":  "Positive",
}

RISK_SIGNALS_POSITIVE = [
    "award or recognition",
    "product launch success",
    "positive customer feedback",
    "strong financial results",
    "community support",
    "innovation praised",
]

RISK_SIGNALS_NEGATIVE = [
    "safety recall or product defect",
    "data breach or privacy violation",
    "executive misconduct or scandal",
    "regulatory fine or legal action",
    "environmental damage",
    "financial fraud or loss",
    "employee mistreatment",
    "whistleblower allegations",
]

RISK_SIGNALS_NEUTRAL = [
    "industry trend",
    "market change",
    "company restructuring",
    "partnership announcement",
    "leadership transition",
]

ALL_SIGNALS = RISK_SIGNALS_NEGATIVE + RISK_SIGNALS_POSITIVE + RISK_SIGNALS_NEUTRAL

URGENCY_LABELS = ["requires immediate response", "can wait a few days", "low priority monitoring"]

DRAFT_TEMPLATES = {
    "Critical": (
        "We are aware of the serious concerns raised in recent reports and want to address them directly. "
        "The safety and trust of our customers and stakeholders is our highest priority, and we take these "
        "allegations with the utmost seriousness. We have launched an immediate internal investigation and "
        "are cooperating fully with relevant authorities. We are committed to full transparency and will "
        "provide updates as our review progresses. We sincerely apologize to all those affected and will "
        "not rest until this matter is fully resolved."
    ),
    "High": (
        "We are aware of the concerns that have been raised and want to respond swiftly and openly. "
        "We are actively investigating the matters described and take full responsibility for ensuring "
        "the highest standards in everything we do. We will be providing a detailed update to all "
        "stakeholders within 48 hours. Our leadership team is personally overseeing this situation "
        "and we remain committed to doing what is right."
    ),
    "Medium": (
        "Thank you to everyone who has shared feedback and raised questions about this matter. "
        "We want to be clear that we take all concerns seriously and are reviewing the situation carefully. "
        "We are committed to operating with integrity and will share further information once our "
        "internal review is complete. We appreciate your patience and continued trust in us."
    ),
    "Low": (
        "We appreciate the attention this topic has received and welcome the opportunity to share our perspective. "
        "We are confident in our practices and remain committed to transparency with all of our stakeholders. "
        "We encourage anyone with questions or concerns to reach out directly, and we will continue to "
        "engage openly with our community."
    ),
}


def score_to_label(score: float) -> str:
    """Convert a -1 to 1 sentiment float to a display label."""
    if score > 0.2:
        return "Positive"
    if score < -0.2:
        return "Negative"
    return "Neutral"


def map_sentiment(label: str, score_raw: float) -> tuple[str, int]:
    """Return (display_label, integer_score -100..100)."""
    display = SENTIMENT_LABEL_MAP.get(label.lower(), SENTIMENT_LABEL_MAP.get(label, "Neutral"))
    if display == "Positive":
        int_score = int(score_raw * 100)
    elif display == "Negative":
        int_score = -int(score_raw * 100)
    else:
        int_score = 0
    return display, max(-100, min(100, int_score))


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    # Truncate very long inputs to avoid OOM
    truncated = text[:2000]

    # --- 1. Sentiment ---
    sent_pipe = get_sentiment_pipeline()
    sent_result = sent_pipe(truncated)[0]
    sentiment_label, sentiment_score = map_sentiment(sent_result["label"], sent_result["score"])

    # --- 2. Zero-shot: urgency ---
    zs_pipe = get_zero_shot_pipeline()
    urgency_result = zs_pipe(truncated, candidate_labels=URGENCY_LABELS, multi_label=False)
    top_urgency = urgency_result["labels"][0]
    if "immediate" in top_urgency:
        urgency = "Critical" if sentiment_score < -60 else "High"
    elif "few days" in top_urgency:
        urgency = "Medium"
    else:
        urgency = "Low"

    # --- 3. Zero-shot: signal detection ---
    signal_result = zs_pipe(truncated, candidate_labels=ALL_SIGNALS, multi_label=True)
    # Pick top 5 signals above a confidence threshold
    threshold = 0.25
    signals = []
    for label, score in zip(signal_result["labels"], signal_result["scores"]):
        if score >= threshold and len(signals) < 5:
            if label in RISK_SIGNALS_POSITIVE:
                sig_type = "positive"
            elif label in RISK_SIGNALS_NEGATIVE:
                sig_type = "negative"
            else:
                sig_type = "neutral"
            # Shorten label for display
            short = " ".join(label.split()[:4]).title()
            signals.append({"label": short, "type": sig_type})

    if not signals:
        signals = [{"label": "General mention", "type": "neutral"}]

    # --- 4. Reputation risk score (0-10) ---
    neg_signal_count = sum(1 for s in signals if s["type"] == "negative")
    base_risk = abs(sentiment_score) / 10          # 0-10 from sentiment
    signal_boost = neg_signal_count * 1.5
    reputation_risk = min(10, round(base_risk * 0.6 + signal_boost))

    urgency_reason_map = {
        "Critical": "Immediate public and media attention demands a rapid, coordinated response.",
        "High":     "Escalating coverage suggests a response window of 24-48 hours.",
        "Medium":   "Developing story — monitor closely and prepare a holding statement.",
        "Low":      "Situation is stable; standard monitoring cadence is sufficient.",
    }
    urgency_reason = urgency_reason_map[urgency]

    # --- 5. Summarization ---
    summarizer = get_summarization_pipeline()
    input_len = len(truncated.split())
    max_sum = min(120, max(40, input_len // 2))
    min_sum = min(40, max_sum - 10)
    summary_result = summarizer(truncated, max_length=max_sum, min_length=min_sum, do_sample=False)
    summary = summary_result[0]["summary_text"]

    # --- 6. Draft response (template-based, reliable across all hardware) ---
    draft_response = DRAFT_TEMPLATES[urgency]

    return {
        "sentiment": sentiment_label,
        "sentimentScore": sentiment_score,
        "reputationRisk": reputation_risk,
        "urgency": urgency,
        "urgencyReason": urgency_reason,
        "signals": signals,
        "summary": summary,
        "draftResponse": draft_response,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
