from __future__ import annotations

import pytest

import rag_llm


pytestmark = pytest.mark.integration


CLARIFICATION_CASES = [
    pytest.param(
        "Bir dönemde kaç AKTS alabilirim?",
        "term_scope",
        id="akts-term-scope-clarification",
    ),
    pytest.param(
        "Devamsızlık sınırı nedir?",
        None,
        id="devamsizlik-clarification",
    ),
    pytest.param(
    "Sınav hakkı kaç tane?",
    None,
    id="sinav-hakki-clarification",
),
]


@pytest.fixture(scope="module", autouse=True)
def initialize_real_rag() -> None:
    rag_llm.initialize_rag()


@pytest.mark.parametrize(
    "question, expected_clarification_type",
    CLARIFICATION_CASES,
)
def test_ambiguous_question_requests_expected_clarification(
    question: str,
    expected_clarification_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Eğer sistem clarification vermeyip LLM'e kadar giderse
    # gerçek Ollama çağrısı yapmak yerine kontrollü bir cevap döndür.
    monkeypatch.setattr(
        rag_llm.ollama,
        "chat",
        lambda **kwargs: {
            "message": {
                "content": "__LLM_GENERATION_CALLED__",
            }
        },
    )

    answer, _, _, _ = rag_llm.ask_with_clarification(
        question=question,
        clarification=None,
        history=[],
        top_k=12,
        allow_generic_rewrite=True,
    )

    if expected_clarification_type is None:
        expected_prefix = f"{rag_llm.CLARIFY_PREFIX}|"
    else:
        expected_prefix = (
        f"{rag_llm.CLARIFY_PREFIX}|"
        f"{expected_clarification_type}|"
    )

    assert answer.startswith(expected_prefix), (
        "Sistem beklenen clarification yerine başka bir davranış üretti. "
        f"Soru: {question!r}, "
        f"beklenen tip: {expected_clarification_type!r}, "
        f"gerçek çıktı: {answer!r}"
    )

def test_explicit_exam_type_does_not_request_generic_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag_llm.ollama,
        "chat",
        lambda **kwargs: {
            "message": {
                "content": "__LLM_GENERATION_CALLED__",
            }
        },
    )

    answer, _, _, _ = rag_llm.ask_with_clarification(
        question="Ek sınav hakkı kaç tane?",
        clarification=None,
        history=[],
        top_k=12,
        allow_generic_rewrite=True,
    )

    assert not answer.startswith(rag_llm.CLARIFY_PREFIX)


def test_explicit_special_student_question_does_not_request_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag_llm.ollama,
        "chat",
        lambda **kwargs: {
            "message": {
                "content": "__LLM_GENERATION_CALLED__",
            }
        },
    )

    answer, _, _, _ = rag_llm.ask_with_clarification(
        question="Özel öğrenci kredi toplamı nasıl değerlendirilir?",
        clarification=None,
        history=[],
        top_k=12,
        allow_generic_rewrite=True,
    )

    assert not answer.startswith(rag_llm.CLARIFY_PREFIX)