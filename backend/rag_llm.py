from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchText, MatchValue
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("LLM_MODEL", "gemma3:4b")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "mevzuat_rag")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "ytu-ce-cosmos/turkish-e5-large")

SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.45"))
TOP_K = int(os.getenv("TOP_K", "12"))
FALLBACK_TOP_K = int(os.getenv("FALLBACK_TOP_K", "18"))

CLARIFY_PREFIX = "_CLARIFY_"

LOW_CONF_TOP_SCORE_THRESHOLD = float(os.getenv("LOW_CONF_TOP_SCORE_THRESHOLD", "0.58"))
LOW_CONF_TOP_GAP_THRESHOLD = float(os.getenv("LOW_CONF_TOP_GAP_THRESHOLD", "0.03"))
LOW_CONF_CANDIDATE_SPREAD_THRESHOLD = float(os.getenv("LOW_CONF_CANDIDATE_SPREAD_THRESHOLD", "0.04"))

embed_model: Optional[SentenceTransformer] = None
client: Optional[QdrantClient] = None

logger = logging.getLogger(__name__)

FACET_VALUE_LABELS: Dict[str, Dict[str, str]] = {
    "term_scope": {
        "normal_donem": "Normal dönem",
        "yaz_okulu": "Yaz öğretimi",
    },
    "program_level": {
        "onlisans": "Önlisans",
        "lisans": "Lisans",
        "yuksek_lisans": "Yüksek lisans",
        "doktora": "Doktora",
        "lisansustu": "Lisansüstü",
    },
    "student_status": {
        "normal": "Normal öğrenci",
        "ozel_ogrenci": "Özel öğrenci",
        "uluslararasi_ogrenci": "Uluslararası öğrenci",
        "cift_anadal_yandal": "Çift anadal / yandal öğrencisi",
    },
    "teaching_mode": {
        "orgun": "Örgün öğretim",
        "uzaktan": "Uzaktan öğretim",
    },
    "topic_domain": {
        "yaz_ogretimi": "Yaz öğretimi",
        "uzaktan_ogretim": "Uzaktan öğretim",
        "ozel_ogrenci": "Özel öğrenci",
        "uluslararasi_ogrenci": "Uluslararası öğrenci",
        "cift_anadal_yandal": "Çift anadal / yandal",
        "yatay_gecis_intibak": "Yatay geçiş / intibak / muafiyet",
        "staj_uygulamali": "Staj / uygulamalı eğitim",
        "disiplin": "Disiplin",
        "akademik_danismanlik": "Akademik danışmanlık",
        "yabanci_dil": "Yabancı dil / hazırlık",
        "ek_sinav": "Ek sınav / azami süre",
        "ders_yuku": "Ders yükü / AKTS",
        "general": "Genel",
    },
    "topic": {
        "akts": "AKTS",
        "ders_yuku": "Ders yükü",
        "mezuniyet": "Mezuniyet",
        "sinav": "Sınav",
        "devamsizlik": "Devamsızlık",
        "kayit": "Kayıt",
        "staj": "Staj",
        "unknown": "Genel konu",
    },
}

FOLLOWUP_START_HINTS = (
    "peki",
    "ya",
    "bu",
    "bunda",
    "bunu",
    "bunlar",
    "onun",
    "o zaman",
    "hangisi",
)

QUESTION_PATTERNS: Dict[str, List[Tuple[str, re.Pattern]]] = {
    "term_scope": [
        ("yaz_okulu", re.compile(r"\b(yaz\s*okulu|yaz\s*öğretimi|yaz\s*ogretimi)\b", re.IGNORECASE)),
        (
            "normal_donem",
            re.compile(
                r"\b(normal\s*dönem|normal\s*donem|güz\s*dönemi|guz\s*donemi|bahar\s*dönemi|bahar\s*donemi|yarıyıl|yariyil)\b",
                re.IGNORECASE,
            ),
        ),
    ],
    "program_level": [
        ("onlisans", re.compile(r"\b(ön\s*lisans|on\s*lisans)\b", re.IGNORECASE)),
        ("yuksek_lisans", re.compile(r"\b(yüksek\s*lisans|yuksek\s*lisans|master)\b", re.IGNORECASE)),
        ("doktora", re.compile(r"\b(doktora|phd|ph\.d)\b", re.IGNORECASE)),
        ("lisansustu", re.compile(r"\b(lisansüstü|lisansustu|lisans\s*üstü)\b", re.IGNORECASE)),
        ("lisans", re.compile(r"\blisans\b", re.IGNORECASE)),
    ],
    "student_status": [
        ("ozel_ogrenci", re.compile(r"\b(özel\s*öğrenci|ozel\s*ogrenci)\b", re.IGNORECASE)),
        (
            "uluslararasi_ogrenci",
            re.compile(r"\b(uluslararası\s*öğrenci|uluslararasi\s*ogrenci|yabancı\s*uyruklu|yabanci\s*uyruklu)\b", re.IGNORECASE),
        ),
        ("cift_anadal_yandal", re.compile(r"\b(çift\s*anadal|cift\s*anadal|yandal|yan\s*dal)\b", re.IGNORECASE)),
    ],
    "teaching_mode": [
        ("uzaktan", re.compile(r"\b(uzaktan|online|çevrimiçi|cevrimici|harmanlanmış|harmanlanmis|karma)\b", re.IGNORECASE)),
        ("orgun", re.compile(r"\b(örgün|orgun|yüz\s*yüze|yuz\s*yuze)\b", re.IGNORECASE)),
    ],
}

DOMAIN_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("yaz_ogretimi", re.compile(r"\b(yaz\s*okulu|yaz\s*öğretimi|yaz\s*ogretimi)\b", re.IGNORECASE)),
    ("uzaktan_ogretim", re.compile(r"\b(uzaktan|online|çevrimiçi|cevrimici|harmanlanmış|harmanlanmis|karma)\b", re.IGNORECASE)),
    ("ozel_ogrenci", re.compile(r"\b(özel\s*öğrenci|ozel\s*ogrenci)\b", re.IGNORECASE)),
    (
        "uluslararasi_ogrenci",
        re.compile(r"\b(uluslararası\s*öğrenci|uluslararasi\s*ogrenci|yabancı\s*uyruklu|yabanci\s*uyruklu)\b", re.IGNORECASE),
    ),
    ("cift_anadal_yandal", re.compile(r"\b(çift\s*anadal|cift\s*anadal|yandal|yan\s*dal)\b", re.IGNORECASE)),
    ("yatay_gecis_intibak", re.compile(r"\b(yatay\s*geçiş|yatay\s*gecis|muafiyet|intibak)\b", re.IGNORECASE)),
    (
        "staj_uygulamali",
        re.compile(r"\b(staj|uygulamalı\s*eğitim|uygulamali\s*egitim|iş\s*yeri\s*uygulaması|is\s*yeri\s*uygulamasi|ume)\b", re.IGNORECASE),
    ),
    ("disiplin", re.compile(r"\b(disiplin|uyarma|kınama|kinama|uzaklaştırma|uzaklastirma)\b", re.IGNORECASE)),
    ("yabanci_dil", re.compile(r"\b(yabancı\s*dil|yabanci\s*dil|hazırlık|hazirlik)\b", re.IGNORECASE)),
    ("ek_sinav", re.compile(r"\b(ek\s*sınav|ek\s*sinav|azami\s*süre|azami\s*sure)\b", re.IGNORECASE)),
    ("ders_yuku", re.compile(r"\b(AKTS|ders\s*yükü|ders\s*yuku|ders\s*alma|toplam\s*kredi|alınabilecek\s*kredi|alinabilecek\s*kredi)\b", re.IGNORECASE)),
]

TURKISH_STOPWORDS = {
    "ve", "veya", "ile", "için", "icin", "mi", "mı", "mu", "mü", "bu", "şu", "su", "bir",
    "de", "da", "ne", "nedir", "nasıl", "nasil", "kaç", "kac", "neye", "göre", "gore",
    "olan", "olarak", "hangi", "ait", "gibi", "ki", "ya", "peki"
}

NO_ANSWER_MESSAGES = (
    "mevzuatta bu konuyla ilgili bilgi bulamadım.",
    "bu soruya ilişkin mevzuatta net bir hüküm bulunmamaktadır.",
    "mevzuatta bu konuda açık hüküm yok.",
)


@dataclass
class RankedHit:
    payload: Dict[str, Any]
    score: float
    dense_score: float
    lexical_score: float
    rerank_score: float


def initialize_rag() -> None:
    global embed_model, client

    if embed_model is not None and client is not None:
        return

    logger.info("Embedding modeli yükleniyor: %s", EMBED_MODEL)
    embed_model = SentenceTransformer(EMBED_MODEL)

    logger.info("Qdrant bağlantısı yapılıyor: %s", QDRANT_URL)
    client = QdrantClient(url=QDRANT_URL)

    logger.info("RAG sistemi hazır. LLM=%s", MODEL_NAME)


def get_value_label(facet: str, value: str) -> str:
    return FACET_VALUE_LABELS.get(facet, {}).get(value, value.replace("_", " ").title())


def is_no_answer_text(answer: str) -> bool:
    normalized = _normalize_text(answer)

    if any(
        _normalize_text(message) == normalized
        for message in NO_ANSWER_MESSAGES
    ):
        return True

    for prefix in ("hayır ", "hayir "):
        if normalized.startswith(prefix):
            remainder = normalized[len(prefix):].strip()

            if any(
                _normalize_text(message) == remainder
                for message in NO_ANSWER_MESSAGES
            ):
                return True

    return False

def _normalize_text(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^\wçğıöşü\s]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize_content(text: str) -> List[str]:
    normalized = _normalize_text(text)
    tokens = []
    for token in normalized.split():
        if len(token) <= 1:
            continue
        if token in TURKISH_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _is_explicit_independent_question(question: str) -> bool:
    q = _normalize_text(question)
    tokens = q.split()
    if not tokens:
        return False
    if "?" in question:
        return True
    if _infer_question_domain(q) != "general":
        return True
    if len(tokens) >= 5 and any(x in q for x in ["nedir", "nasıl", "ne zaman", "kaç", "hangi", "şart", "süresi", "ücreti", "puan"]):
        return True
    return False


def _looks_like_followup(question: str) -> bool:
    q = _normalize_text(question)
    tokens = q.split()

    if not tokens:
        return False

    if q.startswith(FOLLOWUP_START_HINTS):
        return True

    # çok kısa ve açık domain taşımıyorsa follow-up olabilir
    if len(tokens) <= 2 and _infer_question_domain(q) == "general":
        return True

    # açık bağımsız soruysa önceki soruya bağlama
    if _is_explicit_independent_question(question):
        return False

    if _infer_question_domain(q) != "general":
        return False

    return False


def _recent_history(history: List[Dict[str, str]], max_items: int = 4) -> List[Dict[str, str]]:
    if not history:
        return []
    return history[-max_items:]


def _should_use_generation_history(question: str, history: List[Dict[str, str]], filter_params: Optional[Dict[str, Any]]) -> bool:
    if not history:
        return False
    if filter_params:
        return True
    return _looks_like_followup(question)


def _build_search_text(question: str, history: List[Dict[str, str]]) -> str:
    if not history or not _looks_like_followup(question):
        return question

    for msg in reversed(history):
        if msg.get("role") == "user":
            prev_q = (msg.get("content") or "").strip()
            if prev_q:
                return f"{prev_q} {question}"
            break

    return question


def _infer_question_domain(question: str) -> str:
    for domain, pattern in DOMAIN_PATTERNS:
        if pattern.search(question):
            return domain
    return "general"


def _extract_explicit_facets(question: str) -> Dict[str, str]:
    result: Dict[str, str] = {}

    for facet, rules in QUESTION_PATTERNS.items():
        for value, pattern in rules:
            if pattern.search(question):
                result[facet] = value
                break

    if result.get("program_level") == "lisans":
        q = question.lower()
        if "yüksek lisans" in q or "yuksek lisans" in q or "lisansüstü" in q or "lisansustu" in q:
            result.pop("program_level", None)

    return result


def _derive_missing_payload_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(payload)
    source = str(payload.get("source", "") or "").lower()
    text = str(payload.get("text", "") or "").lower()
    haystack = f"{source}\n{text}"

    if not enriched.get("term_scope") or enriched.get("term_scope") == "unknown":
        if "yaz öğret" in haystack or "yaz ogret" in haystack or "yaz okulu" in haystack:
            enriched["term_scope"] = "yaz_okulu"

    if not enriched.get("student_status") or enriched.get("student_status") in ("unknown", None):
        if "özel öğrenci" in haystack or "ozel ogrenci" in haystack:
            enriched["student_status"] = "ozel_ogrenci"
        elif any(x in haystack for x in ["uluslararası öğrenci", "uluslararasi ogrenci", "yabancı uyruklu", "yabanci uyruklu"]):
            enriched["student_status"] = "uluslararasi_ogrenci"
        elif any(x in haystack for x in ["çift anadal", "cift anadal", "yandal", "yan dal"]):
            enriched["student_status"] = "cift_anadal_yandal"
        else:
            enriched["student_status"] = "normal"

    if not enriched.get("teaching_mode") or enriched.get("teaching_mode") == "unknown":
        if "uzaktan öğretim" in haystack or "uzaktan ogretim" in haystack:
            enriched["teaching_mode"] = "uzaktan"

    if not enriched.get("topic_domain") or enriched.get("topic_domain") == "general":
        enriched["topic_domain"] = _infer_question_domain(haystack)

    return enriched


def _build_filter(filter_params: Optional[Dict[str, Any]]) -> Optional[Filter]:
    if not filter_params:
        return None

    must_conditions = []
    for key, value in filter_params.items():
        must_conditions.append(
            FieldCondition(
                key=key,
                match=MatchValue(value=value),
            )
        )

    return Filter(must=must_conditions) if must_conditions else None


def _retrieve_dense_hits(
    question: str,
    history: List[Dict[str, str]],
    top_k: int,
    filter_params: Optional[Dict[str, Any]],
    use_history: bool = True,
) -> List[Any]:
    assert embed_model is not None
    assert client is not None

    effective_history = history if use_history else []
    search_text = _build_search_text(question, effective_history)
    query_embedding = embed_model.encode(
        f"query: {search_text}",
        normalize_embeddings=True,
    )

    query_filter = _build_filter(filter_params)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        query_filter=query_filter,
        limit=top_k,
    )

    return response.points or []

def _extract_definition_term(question: str) -> Optional[str]:
    q = question.strip()

    patterns = (
        r"^(.{1,50}?)\s+kısaltması\s+ne\s+anlama\s+gelir\??$",
        r"^(.{1,50}?)\s+açılımı\s+nedir\??$",
        r"^(.{1,50}?)\s+ne\s+demektir\??$",
        r"^([\wÇĞİÖŞÜçğıöşü\-]{2,40})\s+nedir\??$",
    )

    for pattern in patterns:
        match = re.match(pattern, q, re.IGNORECASE)

        if match:
            return match.group(1).strip(" '\"“”")

    return None


def _retrieve_definition_hits(
    question: str,
    top_k: int,
    filter_params: Optional[Dict[str, Any]],
) -> List[Any]:
    term = _extract_definition_term(question)

    if not term:
        return []

    assert embed_model is not None
    assert client is not None

    query_embedding = embed_model.encode(
        f"query: Tanımlar {term}",
        normalize_embeddings=True,
    )

    base_filter = _build_filter(filter_params)
    must_conditions = list(base_filter.must or []) if base_filter else []

    must_conditions.append(
        FieldCondition(
            key="text",
            match=MatchText(text=term),
        )
    )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        query_filter=Filter(must=must_conditions),
        limit=max(top_k, 50),
    )

    return response.points or []

def _has_explicit_definition_candidate(
    question: str,
    hits: List[RankedHit],
) -> bool:
    term = _extract_definition_term(question)

    if not term:
        return False

    for hit in hits:
        payload = hit.payload or {}

        raw_text = " ".join(
            [
                str(payload.get("madde_header", "") or ""),
                str(payload.get("madde", "") or ""),
                str(payload.get("text", "") or ""),
            ]
        )

        if re.search(
            rf"(?<!\w){re.escape(term)}\s*:",
            raw_text,
            re.IGNORECASE,
        ):
            return True

    return False

def _lexical_score(question: str, payload: Dict[str, Any]) -> float:
    question_norm = _normalize_text(question)
    query_tokens = set(_tokenize_content(question))
    if not query_tokens:
        return 0.0

    text_parts = [
        str(payload.get("source", "") or ""),
        str(payload.get("madde_header", "") or ""),
        str(payload.get("madde", "") or ""),
        str(payload.get("text", "") or ""),
        str(payload.get("topic_domain", "") or ""),
        str(payload.get("topic", "") or ""),
        str(payload.get("student_status", "") or ""),
    ]
    haystack = " ".join(text_parts)
    haystack_norm = _normalize_text(haystack)
    haystack_tokens = set(_tokenize_content(haystack))

    matched = len(query_tokens & haystack_tokens)
    coverage = matched / max(len(query_tokens), 1)

    phrase_bonus = 0.0

    if (
        question_norm
        and len(question_norm.split()) >= 3
        and question_norm in haystack_norm
):
        phrase_bonus += 0.12


    bigrams = []
    q_words = question_norm.split()
    for i in range(len(q_words) - 1):
        bigrams.append(f"{q_words[i]} {q_words[i + 1]}")
    if any(bg in haystack_norm for bg in bigrams):
        phrase_bonus += 0.06

    explicit_domain = _infer_question_domain(question_norm)
    domain_bonus = 0.0
    topic_domain = str(payload.get("topic_domain", "") or "")
    student_status = str(payload.get("student_status", "") or "")
    topic = str(payload.get("topic", "") or "")

    if explicit_domain != "general":
        if topic_domain == explicit_domain:
            domain_bonus += 0.12
        elif explicit_domain == "cift_anadal_yandal" and student_status == "cift_anadal_yandal":
            domain_bonus += 0.10
        else:
            domain_bonus -= 0.08

    explicit_facets = _extract_explicit_facets(question_norm)
    facet_bonus = 0.0
    for facet_name, facet_value in explicit_facets.items():
        if str(payload.get(facet_name, "") or "") == facet_value:
            facet_bonus += 0.04

    if _is_quantity_question(question_norm) and payload.get("contains_akts"):
        facet_bonus += 0.05

    if "staj" in query_tokens and topic == "staj":
        facet_bonus += 0.05

    raw_score = coverage + phrase_bonus + domain_bonus + facet_bonus
    return max(0.0, min(raw_score, 1.0))


def _rerank_hits(question: str, hits: List[Any]) -> List[RankedHit]:
    ranked: List[RankedHit] = []

    for hit in hits:
        payload = _derive_missing_payload_fields(hit.payload or {})
        dense_score = float(hit.score or 0.0)
        lexical_score = _lexical_score(question, payload)
        rerank_score = (0.72 * dense_score) + (0.28 * lexical_score)
        definition_term = _extract_definition_term(question)

        if definition_term:
            raw_text = " ".join(
                [
                    str(payload.get("madde_header", "") or ""),
                    str(payload.get("madde", "") or ""),
                    str(payload.get("text", "") or ""),
                ]
            )

            if re.search(
                rf"(?<!\w){re.escape(definition_term)}\s*:",
                raw_text,
                re.IGNORECASE,
            ):
                rerank_score += 0.20

        rerank_score = max(0.0, min(rerank_score, 1.0))

        ranked.append(
            RankedHit(
                payload=payload,
                score=rerank_score,
                dense_score=dense_score,
                lexical_score=lexical_score,
                rerank_score=rerank_score,
            )
        )

    ranked.sort(key=lambda h: (h.rerank_score, h.dense_score, h.lexical_score), reverse=True)
    return ranked


def _retrieve(
    question: str,
    history: List[Dict[str, str]],
    top_k: int,
    filter_params: Optional[Dict[str, Any]],
    use_history: bool = True,
) -> List[RankedHit]:
    dense_hits = _retrieve_dense_hits(
        question,
        history,
        top_k=top_k,
        filter_params=filter_params,
        use_history=use_history,
)

    definition_hits = _retrieve_definition_hits(
        question=question,
        top_k=top_k,
        filter_params=filter_params,
)

    combined_hits = list(dense_hits)
    seen_ids = {str(hit.id) for hit in combined_hits}

    for hit in definition_hits:
        hit_id = str(hit.id)

        if hit_id in seen_ids:
            continue

        seen_ids.add(hit_id)
        combined_hits.append(hit)

    return _rerank_hits(question, combined_hits)


def _best_score(hits: List[RankedHit]) -> float:
    return float(hits[0].score or 0.0) if hits else 0.0


def _maybe_upgrade_with_fallback(
    question: str,
    history: List[Dict[str, str]],
    filter_params: Optional[Dict[str, Any]],
    hits: List[RankedHit],
) -> List[RankedHit]:
    """
    İlk arama zayıfsa veya history kaynaklı kirlenme ihtimali varsa,
    history kullanmadan daha geniş bir retrieval denemesi yap.
    """
    explicit_domain = _infer_question_domain(_normalize_text(question))
    initial_top = _best_score(hits)

    need_retry = False
    if not hits:
        need_retry = True
    elif initial_top < max(SCORE_THRESHOLD, 0.54):
        need_retry = True
    elif history and _is_explicit_independent_question(question):
        need_retry = True
    elif explicit_domain != "general" and initial_top < 0.62:
        need_retry = True

    if not need_retry:
        return hits

    fallback_hits = _retrieve(
        question=question,
        history=[],
        top_k=max(FALLBACK_TOP_K, TOP_K),
        filter_params=filter_params,
        use_history=False,
    )

    fallback_top = _best_score(fallback_hits)

    if fallback_top > initial_top + 0.015:
        logger.info(
            "Fallback retrieval promoted | initial=%.4f fallback=%.4f question=%s",
            initial_top,
            fallback_top,
            question,
        )
        return fallback_hits

    if not hits and fallback_hits:
        return fallback_hits

    return hits


def _select_candidate_hits(hits: List[RankedHit]) -> List[RankedHit]:
    if not hits:
        return []

    top_score = float(hits[0].score or 0.0)
    min_score = max(SCORE_THRESHOLD - 0.05, top_score - 0.12)

    candidates = [
        h for h in hits
        if float(h.score or 0.0) >= min_score
    ]

    unique_candidates: List[RankedHit] = []
    seen_texts = set()

    for hit in candidates:
        text = _normalize_text(
            str((hit.payload or {}).get("text", ""))
        )

        if text in seen_texts:
            continue

        seen_texts.add(text)
        unique_candidates.append(hit)

    return unique_candidates[:4]

def _expand_same_article_for_list_question(
    question: str,
    filter_params: Optional[Dict[str, Any]],
    candidates: List[RankedHit],
) -> List[RankedHit]:
    if not candidates:
        return candidates

    q = _normalize_text(question)

    list_signals = (
        "hangi belgeler",
        "görevleri nelerdir",
        "gorevleri nelerdir",
        "hangi durumlarda",
        "şartları nelerdir",
        "sartlari nelerdir",
        "şartlar nelerdir",
        "sartlar nelerdir",
    )

    if not any(signal in q for signal in list_signals):
        return candidates

    anchor = next(
        (
            hit
            for hit in candidates
            if (hit.payload or {}).get("source")
            and (hit.payload or {}).get("madde_no") is not None
        ),
        None,
    )

    if anchor is None:
        return candidates

    anchor_payload = anchor.payload or {}
    anchor_source = anchor_payload.get("source")
    anchor_madde_no = anchor_payload.get("madde_no")

    expanded_hits = _retrieve(
        question=question,
        history=[],
        top_k=30,
        filter_params=filter_params,
        use_history=False,
    )

    result = list(candidates)

    seen = {
        (hit.payload or {}).get("chunk_uid")
        or _normalize_text(str((hit.payload or {}).get("text", "")))
        for hit in result
    }

    for hit in expanded_hits:
        payload = hit.payload or {}

        if (
            payload.get("source") != anchor_source
            or payload.get("madde_no") != anchor_madde_no
        ):
            continue

        identity = (
            payload.get("chunk_uid")
            or _normalize_text(str(payload.get("text", "")))
        )

        if identity in seen:
            continue

        seen.add(identity)
        result.append(hit)

        if len(result) >= 8:
            break

    return result

def _collect_facet_values(hits: List[RankedHit], facet: str) -> List[str]:
    values: List[str] = []
    seen = set()

    for h in hits:
        payload = h.payload or {}
        value = payload.get(facet)

        if not value or value in ("unknown", "mixed", None):
            continue

        if value == "lisans":
            prog_cands = payload.get("program_level_candidates") or []
            if "yuksek_lisans" in prog_cands or "lisansustu" in prog_cands:
                continue

        if value not in seen:
            seen.add(value)
            values.append(value)

    return values


def _collect_payload_values(hits: List[RankedHit], field: str, skip_values: Optional[set] = None) -> List[str]:
    if skip_values is None:
        skip_values = set()

    values: List[str] = []
    seen = set()

    for h in hits:
        payload = h.payload or {}
        value = payload.get(field)

        if value in skip_values or value is None:
            continue

        if value not in seen:
            seen.add(value)
            values.append(value)

    return values


def _is_quantity_question(question: str) -> bool:
    q = question.lower()
    return any(x in q for x in ["kaç", "ne kadar", "akts", "kredi", "ders al", "ders alabilir", "ders alabilirim"])

def _has_specific_query_overlap(
    question: str,
    hits: List[RankedHit],
) -> bool:
    generic_tokens = {
        "üniversite",
        "üniversitenin",
        "ücret",
        "ücreti",
        "kaç",
        "kac",
        "ne",
        "kadar",
        "nedir",
        "nasıl",
        "nasil",
    }

    query_tokens = set(_tokenize_content(question))
    specific_tokens = query_tokens - generic_tokens

    if not specific_tokens:
        return True

    context_tokens = set()

    for hit in hits:
        payload = hit.payload or {}
        context_tokens.update(
            _tokenize_content(
                str(payload.get("text", ""))
            )
        )

    return bool(specific_tokens & context_tokens)

def _has_required_explicit_term_support(
    question: str,
    hits: List[RankedHit],
) -> bool:
    normalized_question = _normalize_text(question)

    explicit_term_groups = [
        ("ücretsiz", "ucretsiz"),
    ]

    for terms in explicit_term_groups:
        if not any(term in normalized_question for term in terms):
            continue

        for hit in hits:
            text = _normalize_text(
                str((hit.payload or {}).get("text", ""))
            )

            if any(term in text for term in terms):
                return True

        return False

    return True

def _build_clarification_message(facet: str, values: List[str]) -> str:
    if facet == "term_scope":
        question = "Bunu normal dönem için mi yoksa yaz öğretimi için mi soruyorsunuz?"
    elif facet == "student_status":
        labels = [get_value_label(facet, v) for v in values]
        question = f"Bunu hangi statü için soruyorsunuz: {', '.join(labels)}?"
    elif facet == "program_level":
        labels = [get_value_label(facet, v) for v in values]
        question = f"Bunu hangi program düzeyi için soruyorsunuz: {', '.join(labels)}?"
    elif facet == "teaching_mode":
        question = "Bunu örgün öğretim için mi yoksa uzaktan öğretim için mi soruyorsunuz?"
    elif facet == "topic_domain":
        labels = [get_value_label(facet, v) for v in values]
        question = f"Sorunuz şu konu alanlarından hangisiyle ilgili: {', '.join(labels)}?"
    elif facet == "topic":
        labels = [get_value_label(facet, v) for v in values]
        question = f"Sorunuz şu konulardan hangisiyle ilgili: {', '.join(labels)}?"
    else:
        labels = [get_value_label(facet, v) for v in values]
        question = f"Bu sorunun cevabı duruma göre değişiyor. Hangisi için soruyorsunuz: {', '.join(labels)}?"

    serialized_values = ",".join(values)
    return f"{CLARIFY_PREFIX}|{facet}|{serialized_values}|{question}"


def _is_under_specified_question(question: str) -> bool:
    q = _normalize_text(question)
    tokens = q.split()
    token_count = len(tokens)

    specificity_signals = [
        "şart", "şartı", "genel not ortalaması", "gno", "başvuru", "başvuru şartı",
        "başvuru tarihi", "ne zaman", "kaç", "ne kadar", "akts", "süre", "süresi",
        "zorunlu", "zorunluluğu", "muafiyet", "intibak", "kurumlar arası",
        "kurumlararası", "stajın", "not ortalaması","açılır", "acilir",
        "hangi durumlarda",
        "görevleri", "gorevleri",
    ]

    has_specific_signal = any(sig in q for sig in specificity_signals) or any(ch.isdigit() for ch in q)

    if token_count <= 2:
        return True
    if token_count <= 4 and not has_specific_signal:
        return True
    if token_count >= 6:
        return False

    return not has_specific_signal


def _build_low_confidence_clarification(field: str, values: List[str]) -> str:
    labels = [get_value_label(field, v) for v in values]
    if field == "topic_domain":
        question = f"Sorunuz birden fazla konuya yakın görünüyor. Şunlardan hangisini kastediyorsunuz: {', '.join(labels)}?"
    else:
        question = f"Sorunuz şu başlıklardan hangisiyle ilgili: {', '.join(labels)}?"
    serialized_values = ",".join(values)
    return f"{CLARIFY_PREFIX}|{field}|{serialized_values}|{question}"


def _build_generic_refine_clarification(question: str, domain: str) -> str:
    
    if domain == "devamsizlik":
        prompt = (
        "Devamsızlık sınırı ders veya program türüne göre değişebiliyor. "
        "Hangi bağlam için soruyorsunuz? "
        "Örneğin teorik ders, uygulamalı/laboratuvar dersi, "
        "yabancı dil hazırlık veya uzaktan öğretim olabilir."
    )
        
    elif domain == "sinav":
        prompt = (
        "Sınav hakkı sınav türüne ve öğrencinin durumuna göre değişebiliyor. "
        "Hangi sınav hakkını soruyorsunuz? "
        "Örneğin ek sınav, mezuniyet sınavı, bütünleme, mazeret sınavı "
        "veya tek ders sınavı olabilir."
    )
    elif domain == "ders_yuku":
    
        prompt = (
            "Sorunuz ders yüküyle ilgili görünüyor ama çok yakın birkaç aday bulundu. "
            "Lütfen sorunuzu biraz daha net yazar mısınız? "
            "Örneğin bir yarıyılda alınabilecek AKTS/kredi, yaz öğretimi, özel öğrenci veya çift anadal/yandal bağlamını belirtebilirsiniz."
        )
    elif domain == "staj_uygulamali":
        prompt = (
            "Sorunuz staj / uygulamalı eğitimle ilgili görünüyor ama çok yakın birkaç aday bulundu. "
            "Lütfen biraz daha net yazar mısınız? "
            "Örneğin stajın zamanı, süresi, zorunluluğu veya başvuru/onay sürecini belirtebilirsiniz."
        )
    elif domain == "yatay_gecis_intibak":
        prompt = (
            "Sorunuz yatay geçiş / intibak / muafiyet alanına yakın görünüyor. "
            "Lütfen biraz daha net yazar mısınız? "
            "Örneğin yatay geçiş, ders muafiyeti veya intibak kısmını belirtin."
        )
    elif domain == "yabanci_dil":
        prompt = (
            "Sorunuz yabancı dil / hazırlık alanına yakın görünüyor. "
            "Lütfen biraz daha net yazar mısınız? "
            "Örneğin muafiyet, hazırlık sınıfı, başarı şartı veya sınavı belirtin."
        )
    elif domain == "ek_sinav":
        prompt = (
            "Sorunuz ek sınav / azami süre alanına yakın görünüyor. "
            "Lütfen biraz daha net yazar mısınız? "
            "Örneğin ek sınav hakkı, azami süre veya başarısız ders durumunu belirtin."
        )
    else:
        prompt = (
            "Sorunuz için birkaç çok yakın mevzuat adayı bulundu. "
            "Daha doğru cevap verebilmem için lütfen sorunuzu biraz daha açık yazar mısınız?"
        )

    return f"{CLARIFY_PREFIX}|rewrite_question||{prompt}"


def _score_metrics(candidates: List[RankedHit]) -> Dict[str, float]:
    top_score = float(candidates[0].score or 0.0)
    second_score = float(candidates[1].score or 0.0) if len(candidates) > 1 else 0.0
    last_score = float(candidates[-1].score or 0.0)

    return {
        "top_score": top_score,
        "second_score": second_score,
        "top_gap": top_score - second_score if len(candidates) > 1 else top_score,
        "spread": top_score - last_score,
    }



def _has_clear_winner(question: str, candidates: List[RankedHit]) -> bool:
    if not candidates:
        return False

    metrics = _score_metrics(candidates)
    explicit_domain = _infer_question_domain(question)

    if explicit_domain == "general":
        return metrics["top_score"] >= 0.58 and metrics["top_gap"] >= 0.035

    return metrics["top_score"] >= 0.62 and metrics["top_gap"] >= 0.03


def _facet_candidates_for_clarification(question: str, hits: List[RankedHit]) -> List[RankedHit]:
    candidates = _select_candidate_hits(hits)
    if not candidates:
        return []

    explicit_domain = _infer_question_domain(question)

    # Genel sorularda candidate domain'leri kullanıcıya sormak çok gürültü üretiyor.
    # Bu yüzden genel sorularda clarification'ı en aza indiriyoruz.
    if explicit_domain == "general":
        return candidates[:2]

    return candidates



def _detect_conflict(question: str, hits: List[RankedHit], filter_params: Optional[Dict[str, Any]]) -> Optional[str]:
    if not hits or filter_params:
        return None

    explicit = _extract_explicit_facets(question)
    explicit_domain = _infer_question_domain(question)

    normalized_question = _normalize_text(question)

    generic_attendance_limit = (
    (
        "devamsızlık" in normalized_question
        or "devamsizlik" in normalized_question
    )
    and (
        "sınır" in normalized_question
        or "sinir" in normalized_question
        or (
            any(
                signal in normalized_question
                for signal in ("kaç", "kac", "ne kadar", "nedir")
            )
            and any(
                signal in normalized_question
                for signal in ("oran", "yüzde", "yuzde")
            )
        )
    )
)

    if (
        explicit_domain == "general"
        and generic_attendance_limit
):
        return _build_generic_refine_clarification(
            question,
            "devamsizlik",
    )

    
    explicit_exam_type = any(
        signal in normalized_question
        for signal in (
        "ek sınav",
        "ek sinav",
        "mezuniyet sınav",
        "mezuniyet sinav",
        "bütünleme",
        "butunleme",
        "mazeret sınav",
        "mazeret sinav",
        "tek ders sınav",
        "tek ders sinav",
        "yarıyıl sonu sınav",
        "yariyil sonu sinav",
        "yılsonu sınav",
        "yilsonu sinav",
        "final",
        "ara sınav",
        "ara sinav",
        "vize",
    )
)

    generic_exam_right = (
        ("sınav" in normalized_question or "sinav" in normalized_question)
        and "hakk" in normalized_question
        and any(
            signal in normalized_question
            for signal in ("kaç", "kac", "ne kadar")
    )
)

    if (
        explicit_domain == "general"
        and generic_exam_right
        and not explicit_exam_type
    ):
        return _build_generic_refine_clarification(
            question,
            "sinav",
        )

    generic_term_reference = bool(
        re.search(
            r"\bd[öo]nem(?:de)?\b",
            normalized_question,
            re.IGNORECASE,
        )
    )


    if (
        explicit_domain == "ders_yuku"
        and _is_quantity_question(question)
        and "term_scope" not in explicit
        and generic_term_reference
):
        return _build_clarification_message(
            "term_scope",
            ["normal_donem", "yaz_okulu"],
    )

    candidates = _facet_candidates_for_clarification(question, hits)

    if len(candidates) < 2:
        return None

    # Açık bir kazanan varsa clarification sorma; direkt cevap ver.
    if _has_clear_winner(question, candidates):
        return None

    # Genel sorularda candidate domain/topic üretmek çok gürültülü.
    # Bu yüzden yalnızca açık domain'i olan sorularda ayırıcı facet sor.
    if explicit_domain == "general":
        return None

    facet_priority = ["term_scope", "student_status", "program_level", "teaching_mode"]

    for facet in facet_priority:
        if facet in explicit:
            continue

        values = _collect_facet_values(candidates, facet)
        if len(values) > 1:
            return _build_clarification_message(facet, values)

    if _is_quantity_question(question):
        akts_variants = set()
        for h in candidates:
            payload = h.payload or {}
            akts_vals = tuple(payload.get("akts_values") or [])
            if akts_vals:
                akts_variants.add(akts_vals)

        if len(akts_variants) > 1:
            for facet in ["term_scope", "student_status", "program_level", "teaching_mode"]:
                if facet in explicit:
                    continue
                values = _collect_facet_values(candidates, facet)
                if len(values) > 1:
                    return _build_clarification_message(facet, values)

    return None



def _detect_low_confidence(
    question: str,
    hits: List[RankedHit],
    filter_params: Optional[Dict[str, Any]],
    allow_generic_rewrite: bool = True,
) -> Optional[str]:
    if not hits or filter_params:
        return None

    candidates = _facet_candidates_for_clarification(question, hits)
    if len(candidates) < 2:
        return None

    if _has_explicit_definition_candidate(question, candidates):
        return None

    metrics = _score_metrics(candidates)
    low_confidence = (
        metrics["top_score"] < LOW_CONF_TOP_SCORE_THRESHOLD
        or metrics["top_gap"] < LOW_CONF_TOP_GAP_THRESHOLD
        or metrics["spread"] < LOW_CONF_CANDIDATE_SPREAD_THRESHOLD
    )

    if not low_confidence:
        return None

    logger.info(
        "Low-confidence clarification tetiklendi | rerank_top=%.4f second=%.4f gap=%.4f spread=%.4f",
        metrics["top_score"],
        metrics["second_score"],
        metrics["top_gap"],
        metrics["spread"],
    )

    explicit = _extract_explicit_facets(question)
    explicit_domain = _infer_question_domain(question)

    # Genel sorularda candidate domain/topic sormuyoruz.
    # Gerekirse yalnızca soruyu netleştirmesini istiyoruz.
    if explicit_domain == "general":
        if allow_generic_rewrite and _is_under_specified_question(question):
            return _build_generic_refine_clarification(question, explicit_domain)
        return None

    facet_priority = ["term_scope", "student_status", "program_level", "teaching_mode"]

    for facet in facet_priority:
        if facet in explicit:
            continue
        values = _collect_facet_values(candidates, facet)
        if len(values) > 1:
            return _build_low_confidence_clarification(facet, values)

    if not allow_generic_rewrite:
        return None

    if not _is_under_specified_question(question):
        return None

    return _build_generic_refine_clarification(question, explicit_domain)

def _is_conditional_numeric_question(question: str) -> bool:
    q = _normalize_text(question)

    question_signals = (
        "kaç",
        "kac",
        "ne kadar",
    )

    condition_signals = (
        "en fazla",
        "en az",
        "azami",
        "asgari",
        "kullanmadan",
        "kullanarak",
        "girmeden",
        "şartıyla",
        "sartiyla",
        "durumunda",
        "halinde",
        "kalan",
    )

    target_units = (
        "yarıyıl",
        "yariyil",
        "yıl",
        "yil",
        "ay",
        "gün",
        "gun",
        "ders",
        "kredi",
        "akts",
        "puan",
        "süre",
        "sure",
    )

    return (
        any(signal in q for signal in question_signals)
        and any(signal in q for signal in condition_signals)
        and any(unit in q for unit in target_units)
    )


def _split_numbered_passages(text: str) -> List[str]:
    parts = re.split(
        r"(?m)(?=^\s*(?:"
        r"\d+\s*(?:\\)?[\.\)]"
        r"|\(\d+\)"
        r"|.*?\bMADDE\s+\d+\s*[–—-]"
        r"))",
        text,
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def _focus_numeric_evidence_for_generation(
    question: str,
    candidates: List[RankedHit],
) -> List[RankedHit]:
    if not candidates:
        return candidates

    if not _is_conditional_numeric_question(question):
        return candidates

    question_tokens = set(_tokenize_content(question))
    question_norm = _normalize_text(question)

    condition_signals = (
        "en fazla",
        "en az",
        "azami",
        "asgari",
        "kullanmadan",
        "kullanarak",
        "girmeden",
        "şartıyla",
        "sartiyla",
        "durumunda",
        "halinde",
        "kalan",
    )

    scored_passages = []

    for hit in candidates:
        payload = hit.payload or {}
        text = str(payload.get("text", "") or "")

        for passage in _split_numbered_passages(text):
            passage_tokens = set(_tokenize_content(passage))
            passage_norm = _normalize_text(passage)

            overlap = len(question_tokens & passage_tokens)
            coverage = overlap / max(len(question_tokens), 1)

            soft_matches = 0
            for question_token in question_tokens:
                if question_token in passage_tokens:
                    continue
                if len(question_token) < 5:
                    continue

                if any(
                    len(passage_token) >= 5
                    and question_token[:5] == passage_token[:5]
                    for passage_token in passage_tokens
                ):
                    soft_matches += 1

            soft_overlap_bonus = min(
                0.15,
                soft_matches * 0.03,
            )


            condition_bonus = sum(
                0.06
                for signal in condition_signals
                if signal in question_norm
                and signal in passage_norm
            )

            score = coverage + condition_bonus + soft_overlap_bonus

            scored_passages.append(
                (score, hit, passage)
            )

    if len(scored_passages) < 2:
        return candidates

    scored_passages.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    unique_passages = []

    for score, hit, passage in scored_passages:
        passage_norm = _normalize_text(passage)

        is_overlap_duplicate = any(
            len(passage_norm) >= 80
            and len(existing_norm) >= 80
            and (
                passage_norm in existing_norm
                or existing_norm in passage_norm
            )
            for _, _, _, existing_norm in unique_passages
        )

        if is_overlap_duplicate:
            continue

        unique_passages.append(
            (score, hit, passage, passage_norm)
        )

    if len(unique_passages) < 2:
        return candidates

    best_score, best_hit, best_passage, _ = unique_passages[0]
    second_score = unique_passages[1][0]

    # Belirgin bir kazanan yoksa mevcut context'i koru.
    if (
        best_score < 0.25
        or best_score - second_score < 0.08
    ):
        return candidates

    focused_text = best_passage

    best_hit_text = str(
        (best_hit.payload or {}).get("text", "") or ""
    )

    best_parts = _split_numbered_passages(best_hit_text)
    best_passage_norm = _normalize_text(best_passage)

    for index, part in enumerate(best_parts):
        if _normalize_text(part) != best_passage_norm:
            continue

        if index + 1 < len(best_parts):
            focused_text = (
                f"{part}\n\n"
                f"{best_parts[index + 1]}"
        )

        break

    focused_payload = dict(best_hit.payload or {})
    focused_payload["text"] = focused_text

    return [
        RankedHit(
            payload=focused_payload,
            score=best_hit.score,
            dense_score=best_hit.dense_score,
            lexical_score=best_hit.lexical_score,
            rerank_score=best_hit.rerank_score,
        )
    ]

def _build_context(
hits: List[RankedHit]) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    context_parts: List[str] = []
    sources: List[str] = []
    retrieved: List[Dict[str, Any]] = []

    for h in hits:
        payload = h.payload or {}

        src = payload.get("source", "Bilinmeyen")
        source_url = payload.get("source_url")
        txt = payload.get("text", "")
        madde = payload.get("madde_header") or payload.get("madde", "")
        page_start = payload.get("page_start")

        sources.append(src)

        header = f"KAYNAK: {src}"
        if madde:
            header += f" | MADDE: {madde}"
        if page_start:
            header += f" | SAYFA: {page_start}"

        context_parts.append(f"{header}\nİÇERİK: {txt}")

        retrieved.append(
            {
                "score": round(float(h.score), 4),
                "dense_score": round(float(h.dense_score), 4),
                "lexical_score": round(float(h.lexical_score), 4),
                "source": src,
                "source_url": source_url,
                "madde": madde,
                "page": page_start,
                "term_scope": payload.get("term_scope"),
                "program_level": payload.get("program_level"),
                "student_status": payload.get("student_status"),
                "teaching_mode": payload.get("teaching_mode"),
                "topic_domain": payload.get("topic_domain"),
                "topic": payload.get("topic"),
                "text_preview": (txt or "")[:240],
            }
        )

    unique_sources = sorted(set(sources))
    return "\n\n".join(context_parts), unique_sources, retrieved


def _build_system_prompt() -> str:
    return """Sen KSÜ mevzuat danışmanısın.

Yalnızca sana verilen mevzuat parçalarını kullan.

Her soru için iki davranıştan SADECE BİRİNİ seç:

1. Sorudaki spesifik durum mevzuat parçalarında açıkça destekleniyorsa:
   - Soruyu kısa ve net cevapla.
   - Mevzuatta bulunan madde veya kaynak bilgisini mümkünse belirt.
   - Aşağıdaki sabit ret cümlesini cevap sonuna ASLA ekleme.
   - Soru birden fazla konuyu birlikte soruyorsa ve her konu verilen mevzuat
     parçalarında ayrı ayrı açıkça düzenlenmişse, bu açık hükümleri birlikte
     kullanarak cevap verebilirsin.
   

2. Sorudaki spesifik durum mevzuat parçalarında açıkça desteklenmiyorsa:
   - Yakın veya benzer maddelerden sonuç çıkarma.
   - Yorum, kıyas veya varsayım üretme.
   - Bir şeyin yasak olduğunun yazmaması, onun serbest olduğu anlamına gelmez.
   - Bir şeyin izinli olduğunun yazmaması, onun yasak olduğu anlamına gelmez.
   - Benzer görünen fakat farklı hukuki kavramları birbirinin yerine koyma.
   - Sorudaki işlem, koşul veya statü mevzuatta açıkça aynı şekilde
     düzenlenmiyorsa yakın bir hükümden sonuç çıkarma.
   - Evet/hayır sorularında, cevap vermeden önce mevzuat parçasının sorudaki
     özne, işlem ve koşulu doğrudan desteklediğinden emin ol.
   - Sorudaki işlem yerine yalnızca benzer veya ilişkili başka bir işlem
     düzenleniyorsa evet/hayır sonucu çıkarma.
   - Bu durumda SADECE şu cümleyi yaz:
     Bu soruya ilişkin mevzuatta net bir hüküm bulunmamaktadır.

Bilgiye dayalı bir cevap verdiysen sabit ret cümlesini ayrıca ekleme.
Mevzuatta açıkça desteklenmeyen bilgiyi uydurma."""

def ask(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    top_k: int = TOP_K,
    filter_params: Optional[Dict[str, Any]] = None,
    allow_generic_rewrite: bool = True,
) -> Tuple[str, List[str], float, List[Dict[str, Any]]]:
    if history is None:
        history = []

    if embed_model is None or client is None:
        initialize_rag()

    start_time = time.time()

    effective_filter_params = dict(filter_params or {})

    explicit_facets = _extract_explicit_facets(question)

    if (
    "student_status" not in effective_filter_params
    and explicit_facets.get("student_status") == "ozel_ogrenci"
):
        effective_filter_params["student_status"] = "ozel_ogrenci"

    try:
        hits = _retrieve(question, history, top_k=top_k, filter_params=effective_filter_params, use_history=True)
        hits = _maybe_upgrade_with_fallback(question, history,  effective_filter_params, hits)

        if not hits:
            duration = time.time() - start_time
            return "Mevzuatta bu konuyla ilgili bilgi bulamadım.", [], duration, []

        candidate_hits = _select_candidate_hits(hits)

        candidate_hits = _expand_same_article_for_list_question(
    question,
    effective_filter_params,
    candidate_hits,
)

        question_norm = _normalize_text(question)

        if re.search(
            r"\byabancı\s+dilde\s+(öğretim|eğitim)\b"
            r"|\byabanci\s+dilde\s+(ogretim|egitim)\b",
            question_norm,
            re.IGNORECASE,
        ):
            context_candidates = [
                hit
                for hit in candidate_hits
                if (hit.payload or {}).get("topic_domain")
                != "yabanci_dil_hazirlik"
            ]

            if context_candidates:
                candidate_hits = context_candidates

        if not _has_specific_query_overlap(question, candidate_hits):
            duration = time.time() - start_time
            return (
                "Bu soruya ilişkin mevzuatta net bir hüküm bulunmamaktadır.",
                [],
                duration,
                [],
            )
        
        
        if not _has_required_explicit_term_support(
            question,
            candidate_hits,
        ):
            duration = time.time() - start_time
            return (
                "Bu soruya ilişkin mevzuatta net bir hüküm bulunmamaktadır.",
                [],
                duration,
                [],
            )

        explicit_student_status = explicit_facets.get("student_status")

        is_yes_no_question = bool(
            re.search(
                r"\b(mı|mi|mu|mü|mıdır|midir|mudur|müdür)\b",
                _normalize_text(question),
                re.IGNORECASE,
            )
        )

        special_student_statuses = {
            "cift_anadal_yandal",
            "uluslararasi_ogrenci",
            "ozel_ogrenci",
        }

        if (
            explicit_student_status is None
            and is_yes_no_question
            and candidate_hits
            and all(
                (hit.payload or {}).get("student_status")
                in special_student_statuses
                for hit in candidate_hits
            )
        ):
            duration = time.time() - start_time
            return (
                "Bu soruya ilişkin mevzuatta net bir hüküm bulunmamaktadır.",
                [],
                duration,
                [],
            )       

        facet_clarification = _detect_conflict(question, hits,  effective_filter_params)
        if facet_clarification:
            _, sources, retrieved = _build_context(candidate_hits)
            duration = time.time() - start_time
            return facet_clarification, sources, duration, retrieved

        low_conf_clarification = _detect_low_confidence(
            question,
            hits,
             effective_filter_params,
            allow_generic_rewrite=allow_generic_rewrite,
        )
        if low_conf_clarification:
            _, sources, retrieved = _build_context(candidate_hits)
            duration = time.time() - start_time
            return low_conf_clarification, sources, duration, retrieved

        top_score = float(hits[0].score or 0.0)

        # no-answer kararı öncesi son güvenlik: history'siz geniş aday bakışı zaten yapıldı
        if top_score < SCORE_THRESHOLD:
            duration = time.time() - start_time
            return "Bu soruya ilişkin mevzuatta net bir hüküm bulunmamaktadır.", [], duration, []

        generation_hits = _focus_numeric_evidence_for_generation(
        question,
        candidate_hits,
        )

        context_text, _, _ = _build_context(generation_hits)
        _, unique_sources, retrieved = _build_context(candidate_hits)

        extra_instruction = ""

        if _is_conditional_numeric_question(question):
            extra_instruction = (
                "\n\nEK TALİMAT:\n"
                "Bu soru koşullu ve sayısal bir mevzuat sorusudur. "
                "Verilen ardışık hükümler aynı durumun süresini ve bu sürenin "
                "sonundaki sonucu açıkça düzenliyorsa bu hükümleri birlikte değerlendir. "
                "Süreyi doğrudan süreyi belirleyen hükümden al. "
                "Sorunun farklı ifadelerle aynı koşulu anlatması tek başına ret nedeni değildir. "
                "Bunun dışında benzer veya yakın başka hükümlerden kıyas yapma."
            )
        

        messages: List[Dict[str, str]] = [{"role": "system", "content": _build_system_prompt()}]
        if _should_use_generation_history(question, history, filter_params):
            messages.extend(_recent_history(history, max_items=4))
        messages.append(
            {
                "role": "user",
                "content": (
                f"MEVZUAT BİLGİLERİ:\n{context_text}"
                f"{extra_instruction}"
                f"\n\nSORU: {question}"
),
            }
        )

        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={
                "temperature": 0.0,
                "seed": 42,
},
        )

        answer = response["message"]["content"].strip()
        duration = time.time() - start_time
        return answer, unique_sources, duration, retrieved

    except Exception as e:
        logger.error("RAG hatası: %s", str(e), exc_info=True)
        duration = time.time() - start_time
        return f"Sistem hatası: {str(e)}", [], duration, []


def ask_with_clarification(
    question: str,
    clarification: Optional[Dict[str, str]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    top_k: int = TOP_K,
    allow_generic_rewrite: bool = True,
) -> Tuple[str, List[str], float, List[Dict[str, Any]]]:
    return ask(
        question=question,
        history=history or [],
        top_k=top_k,
        filter_params=clarification,
        allow_generic_rewrite=allow_generic_rewrite,
    )