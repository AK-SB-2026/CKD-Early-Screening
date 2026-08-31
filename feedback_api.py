from datetime import datetime, timezone
from pathlib import Path
import os
import re
import sqlite3
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field

DB_PATH = Path(os.getenv("FEEDBACK_DB_PATH", "feedback.db"))

app = FastAPI(title="RENALIS Feedback API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS feedback (
            submission_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            email TEXT,
            rating INTEGER NOT NULL,
            category TEXT NOT NULL,
            comments TEXT NOT NULL,
            submitted_at TEXT NOT NULL
        )"""
    )
    return conn


class Feedback(BaseModel):
    user_id: str = Field(min_length=4, max_length=32)
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr | None = None
    rating: int = Field(ge=1, le=5)
    category: str = Field(min_length=1, max_length=100)
    comments: str = Field(min_length=1, max_length=5000)


@app.get("/health")
def health():
    return {"status": "ok", "service": "RENALIS Feedback API"}


@app.post("/feedback", status_code=201)
def submit_feedback(feedback: Feedback):
    user_id = feedback.user_id.strip().upper()
    name = re.sub(r"\s+", " ", feedback.name.strip())
    comments = feedback.comments.strip()
    category = feedback.category.strip()

    if not user_id.startswith("REN-"):
        raise HTTPException(status_code=400, detail="Invalid RENALIS ID")

    if not name or not comments:
        raise HTTPException(status_code=400, detail="Name and feedback are required")

    submission_id = "FDB-" + uuid.uuid4().hex[:12].upper()
    submitted_at = datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO feedback
            (submission_id, user_id, name, email, rating, category, comments, submitted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                submission_id,
                user_id,
                name,
                str(feedback.email) if feedback.email else None,
                feedback.rating,
                category,
                comments,
                submitted_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return {"success": True, "submission_id": submission_id}
