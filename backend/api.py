from __future__ import annotations

import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import rag_llm

logger = logging.getLogger(__name__)

sessions: Dict[str, Dict[str, Any]] = {}


def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "pending_clarification": None,
            "original_question": None,
        }
    return sessions[session_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("API başlatılıyor.")
    try:
        rag_llm.initialize_rag()
        logger.info("RAG sistemi hazır.")
    except Exception as e:
        logger.error("Başlangıç hatası: %s", e, exc_info=True)
    yield
    logger.info("API kapanıyor.")


app = FastAPI(title="KSÜ Mevzuat RAG API", version="2.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/", response_model=Dict[str, str])
def read_root():
    return {"message": "KSÜ Mevzuat RAG API Çalışıyor!", "version": "2.2.0"}


@app.get("/health", response_model=HealthResponse)
def health_check():
    rag_ready = rag_llm.embed_model is not None and rag_llm.client is not None
    return HealthResponse(
        status="healthy" if rag_ready else "degraded",
        rag_ready=rag_ready,
        model=rag_llm.MODEL_NAME,
        collection=rag_llm.COLLECTION_NAME,
    )


def _parse_clarification_marker(answer: str) -> Optional[Dict[str, Any]]:
    prefix = f"{rag_llm.CLARIFY_PREFIX}|"
    if not answer.startswith(prefix):
        return None

    try:
        _, ctype, raw_values, question = answer.split("|", 3)
    except ValueError:
        return None

    values = [v for v in raw_values.split(",") if v]
    options = [{"value": v, "label": rag_llm.get_value_label(ctype, v)} for v in values]

    return {
        "type": ctype,
        "question": question,
        "options": options,
    }


def _normalize_text(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^\wçğıöşü\s/.-]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_source_links(retrieved: List[Dict[str, Any]]) -> List[Dict[str, Optional[str]]]:
    links: List[Dict[str, Optional[str]]] = []
    seen = set()

    for item in retrieved:
        source = item.get("source")
        source_url = item.get("source_url")
        key = (source, source_url)
        if not source or key in seen:
            continue
        seen.add(key)
        links.append({"source": source, "url": source_url})

    return links


def _looks_like_new_independent_question(text: str) -> bool:
    normalized = _normalize_text(text)
    tokens = normalized.split()

    if not tokens:
        return False

    if "?" in text:
        return True

    if rag_llm._infer_question_domain(normalized) != "general":
        return True

    if len(tokens) >= 5 and any(x in normalized for x in ["nedir", "nasıl", "ne zaman", "kaç", "şart", "süresi", "ücreti", "puan"]):
        return True

    return False


def _looks_like_pending_option_answer(answer: str, pending: Dict[str, Any]) -> bool:
    answer_lower = _normalize_text(answer)

    for opt in pending.get("options", []):
        value = _normalize_text(opt["value"])
        label = _normalize_text(opt["label"])
        if value and value in answer_lower:
            return True
        if label and label in answer_lower:
            return True

    if pending.get("type") == "rewrite_question":
        return len(answer_lower.split()) <= 8 or not _looks_like_new_independent_question(answer)

    return False


def _parse_clarification_answer(answer: str, pending: Dict[str, Any]) -> Optional[Dict[str, str]]:
    answer_lower = _normalize_text(answer)
    ctype = pending["type"]

    for opt in pending.get("options", []):
        value = _normalize_text(opt["value"])
        label = _normalize_text(opt["label"])

        if value and value in answer_lower:
            return {ctype: opt["value"]}
        if label and label in answer_lower:
            return {ctype: opt["value"]}

    if ctype == "term_scope":
        if "yaz" in answer_lower:
            return {"term_scope": "yaz_okulu"}
        if any(x in answer_lower for x in ["normal", "güz", "guz", "bahar", "yarıyıl", "yariyil"]):
            return {"term_scope": "normal_donem"}

    if ctype == "program_level":
        if "önlisans" in answer_lower or "ön lisans" in answer_lower or "onlisans" in answer_lower:
            return {"program_level": "onlisans"}
        if "yüksek lisans" in answer_lower or "yuksek lisans" in answer_lower or "master" in answer_lower:
            return {"program_level": "yuksek_lisans"}
        if "doktora" in answer_lower or "phd" in answer_lower:
            return {"program_level": "doktora"}
        if "lisans" in answer_lower:
            return {"program_level": "lisans"}

    if ctype == "student_status":
        if "özel" in answer_lower or "ozel" in answer_lower:
            return {"student_status": "ozel_ogrenci"}
        if any(x in answer_lower for x in ["uluslararası", "uluslararasi", "yabancı", "yabanci"]):
            return {"student_status": "uluslararasi_ogrenci"}
        if any(x in answer_lower for x in ["çift anadal", "cift anadal", "yandal", "yan dal"]):
            return {"student_status": "cift_anadal_yandal"}
        if "normal" in answer_lower:
            return {"student_status": "normal"}

    if ctype == "teaching_mode":
        if "uzaktan" in answer_lower or "online" in answer_lower:
            return {"teaching_mode": "uzaktan"}
        if any(x in answer_lower for x in ["örgün", "orgun", "yüz yüze", "yuz yuze"]):
            return {"teaching_mode": "orgun"}

    if ctype == "topic_domain":
        if any(x in answer_lower for x in ["yaz okulu", "yaz öğretimi", "yaz ogretimi"]):
            return {"topic_domain": "yaz_ogretimi"}
        if any(x in answer_lower for x in ["uzaktan", "online", "çevrimiçi", "cevrimici", "karma", "harmanlanmış", "harmanlanmis"]):
            return {"topic_domain": "uzaktan_ogretim"}
        if any(x in answer_lower for x in ["özel öğrenci", "ozel ogrenci"]):
            return {"topic_domain": "ozel_ogrenci"}
        if any(x in answer_lower for x in ["uluslararası", "uluslararasi", "yabancı", "yabanci"]):
            return {"topic_domain": "uluslararasi_ogrenci"}
        if any(x in answer_lower for x in ["çift anadal", "cift anadal", "yandal", "yan dal"]):
            return {"topic_domain": "cift_anadal_yandal"}
        if any(x in answer_lower for x in ["yatay geçiş", "yatay gecis", "muafiyet", "intibak"]):
            return {"topic_domain": "yatay_gecis_intibak"}
        if any(x in answer_lower for x in ["staj", "uygulamalı eğitim", "uygulamali egitim", "iş yeri", "is yeri", "ume"]):
            return {"topic_domain": "staj_uygulamali"}
        if any(x in answer_lower for x in ["disiplin", "uyarma", "kınama", "kinama", "uzaklaştırma", "uzaklastirma"]):
            return {"topic_domain": "disiplin"}
        if any(x in answer_lower for x in ["akademik danışman", "akademik danisman"]):
            return {"topic_domain": "akademik_danismanlik"}
        if any(x in answer_lower for x in ["yabancı dil", "yabanci dil", "hazırlık", "hazirlik"]):
            return {"topic_domain": "yabanci_dil"}
        if any(x in answer_lower for x in ["ek sınav", "ek sinav", "azami süre", "azami sure"]):
            return {"topic_domain": "ek_sinav"}
        if any(x in answer_lower for x in ["akts", "kredi", "ders yükü", "ders yuku", "ders alma"]):
            return {"topic_domain": "ders_yuku"}

    if ctype == "topic":
        if "akts" in answer_lower:
            return {"topic": "akts"}
        if any(x in answer_lower for x in ["kredi", "ders yükü", "ders yuku", "ders alma"]):
            return {"topic": "ders_yuku"}
        if "mezun" in answer_lower:
            return {"topic": "mezuniyet"}
        if any(x in answer_lower for x in ["sınav", "sinav", "vize", "final", "bütünleme", "butunleme", "mazeret"]):
            return {"topic": "sinav"}
        if any(x in answer_lower for x in ["devam", "devamsızlık", "devamsizlik", "yoklama"]):
            return {"topic": "devamsizlik"}
        if any(x in answer_lower for x in ["kayıt", "kayit"]):
            return {"topic": "kayit"}
        if "staj" in answer_lower:
            return {"topic": "staj"}

    return None


@app.post("/ask", response_model=QuestionResponse)
def ask_question(req: QuestionRequest):
    try:
        session_id = req.session_id or str(uuid.uuid4())
        session = get_session(session_id)
        question = req.question.strip()

        filter_params = None
        allow_generic_rewrite = True

        if req.clarification:
            filter_params = req.clarification
            if session.get("original_question"):
                question = session["original_question"]
                session["original_question"] = None
                session["pending_clarification"] = None

        elif session.get("pending_clarification"):
            pending = session["pending_clarification"]

            # Kullanıcı artık yeni, bağımsız bir soru soruyorsa eski clarification'ı iptal et
            if _looks_like_new_independent_question(question) and not _looks_like_pending_option_answer(question, pending):
                session["pending_clarification"] = None
                session["original_question"] = None

            else:
                if pending.get("type") == "rewrite_question":
                    original_question = session.get("original_question") or ""
                    refined_question = req.question.strip()

                    if not refined_question:
                        return QuestionResponse(
                            answer=pending["question"],
                            sources=[],
                            source_links=[],
                            duration=0.0,
                            retrieved=[],
                            session_id=session_id,
                            needs_clarification=True,
                            clarification_options={
                                "type": pending["type"],
                                "message": pending["question"],
                                "options": pending["options"],
                            },
                        )

                    question = f"{original_question} {refined_question}".strip()
                    session["original_question"] = None
                    session["pending_clarification"] = None
                    allow_generic_rewrite = False

                else:
                    parsed = _parse_clarification_answer(question, pending)
                    if parsed:
                        filter_params = parsed
                        question = session["original_question"] or question
                        session["original_question"] = None
                        session["pending_clarification"] = None
                    else:
                        return QuestionResponse(
                            answer=pending["question"],
                            sources=[],
                            source_links=[],
                            duration=0.0,
                            retrieved=[],
                            session_id=session_id,
                            needs_clarification=True,
                            clarification_options={
                                "type": pending["type"],
                                "message": pending["question"],
                                "options": pending["options"],
                            },
                        )

        answer, sources, duration, retrieved = rag_llm.ask(
            question=question,
            history=session["history"],
            filter_params=filter_params,
            allow_generic_rewrite=allow_generic_rewrite,
        )

        needs_clarification = False
        clarification_options = None
        source_links = _build_source_links(retrieved)

        clarification_meta = _parse_clarification_marker(answer)

        if clarification_meta:
            needs_clarification = True
            session["pending_clarification"] = clarification_meta
            session["original_question"] = question

            clarification_options = {
                "type": clarification_meta["type"],
                "message": clarification_meta["question"],
                "options": clarification_meta["options"],
            }

            answer = clarification_meta["question"]
        else:
            # "cevap yok" türü cevapları history'ye yazmıyoruz; yeni soruları kirletmesin
            if not rag_llm.is_no_answer_text(answer):
                session["history"].append({"role": "user", "content": question})
                session["history"].append({"role": "assistant", "content": answer})

        return QuestionResponse(
            answer=answer,
            sources=sources,
            source_links=source_links,
            duration=round(duration, 2),
            retrieved=retrieved,
            session_id=session_id,
            needs_clarification=needs_clarification,
            clarification_options=clarification_options,
        )

    except Exception as e:
        logger.error("API hatası: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sistem hatası: {e}")


@app.post("/ask/clarify", response_model=QuestionResponse)
def ask_with_clarification(req: QuestionRequest):
    if not req.clarification:
        raise HTTPException(status_code=400, detail="clarification parametresi gerekli")
    return ask_question(req)


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} temizlendi"}
    return {"message": "Session bulunamadı"}


@app.get("/session/{session_id}/history")
def get_session_history(session_id: str):
    if session_id not in sessions:
        return {"history": [], "message": "Session bulunamadı"}
    return {"history": sessions[session_id]["history"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)