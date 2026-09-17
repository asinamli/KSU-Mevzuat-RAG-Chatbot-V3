from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Kullanıcı sorusu")
    session_id: Optional[str] = Field(None, description="Oturum ID")
    clarification: Optional[Dict[str, str]] = Field(
        None,
        description="Belirsizlik giderme filtresi",
    )


class QuestionResponse(BaseModel):
    answer: str
    sources: List[str]
    source_links: List[Dict[str, Optional[str]]] = []
    duration: float
    retrieved: List[Dict[str, Any]]
    session_id: str
    needs_clarification: bool = False
    clarification_options: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    rag_ready: bool
    model: str
    collection: str
