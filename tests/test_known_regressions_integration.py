from __future__ import annotations

from click import prompt
import pytest

import rag_llm
import re

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

def test_out_of_scope_parking_fee_returns_no_answer(
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
        question="Üniversitenin otopark abonelik ücreti ne kadar?",
        clarification=None,
        history=[],
        top_k=12,
        allow_generic_rewrite=True,
    )

    assert rag_llm.is_no_answer_text(answer)

def test_summer_term_course_hours_context_contains_expected_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages = {}

    def fake_chat(**kwargs):
        captured_messages["messages"] = kwargs["messages"]
        return {
            "message": {
                "content": "__LLM_GENERATION_CALLED__",
            }
        }

    monkeypatch.setattr(
        rag_llm.ollama,
        "chat",
        fake_chat,
    )

    answer, _, _, _ = rag_llm.ask_with_clarification(
        question="Yaz öğretiminde kaç ders saati alınabilir?",
        clarification=None,
        history=[],
        top_k=12,
        allow_generic_rewrite=True,
    )

    assert answer == "__LLM_GENERATION_CALLED__"

    prompt_text = captured_messages["messages"][-1]["content"].lower()

    assert re.search(
    r"16\s*\(onaltı\s*\)\s*ders\s*saatini",
    prompt_text,
)

    assert "tek ders" in prompt_text

def test_foreign_language_instruction_toefl_context_excludes_preparation_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages = {}

    def fake_chat(**kwargs):
        captured_messages["messages"] = kwargs["messages"]
        return {
            "message": {
                "content": "__LLM_GENERATION_CALLED__",
            }
        }

    monkeypatch.setattr(
        rag_llm.ollama,
        "chat",
        fake_chat,
    )

    question = (
        "Yabanc\u0131 dilde \u00f6\u011fretim i\u00e7in "
        "TOEFL IBT'den minimum ka\u00e7 puan gerekir?"
    )

    answer, _, _, _ = rag_llm.ask_with_clarification(
        question=question,
        clarification=None,
        history=[],
        top_k=12,
        allow_generic_rewrite=True,
    )

    assert answer == "__LLM_GENERATION_CALLED__"

    prompt_text = captured_messages["messages"][-1]["content"]

    assert "TOEFL IBT sınavından 72 puan" in prompt_text
    assert "TOEFL IBT | IELTS" not in prompt_text

def test_two_faculty_registration_without_explicit_rule_returns_no_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm_called = False

    def fake_chat(**kwargs):
        nonlocal llm_called
        llm_called = True
        return {
            "message": {
                "content": "Öğrenciler aynı anda iki farklı fakültede kayıt yaptırabilir."
            }
        }

    monkeypatch.setattr(
        rag_llm.ollama,
        "chat",
        fake_chat,
    )

    question = (
        "\u00dcniversite \u00f6\u011frencileri ayn\u0131 anda iki farkl\u0131 "
        "fak\u00fcltede lisans program\u0131na kay\u0131t yapt\u0131rabilir mi"
    )

    answer, _, _, _ = rag_llm.ask_with_clarification(
        question=question,
        clarification=None,
        history=[],
        top_k=12,
        allow_generic_rewrite=True,
    )

    assert rag_llm.is_no_answer_text(answer)
    assert llm_called is False

def test_free_repeat_claim_without_explicit_support_returns_no_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm_called = False

    def fake_chat(**kwargs):
        nonlocal llm_called
        llm_called = True
        return {
            "message": {
                "content": "Öğrenci başarısız olduğu dersi ücretsiz tekrar alabilir."
            }
        }

    monkeypatch.setattr(rag_llm.ollama, "chat", fake_chat)

    question = (
        "\u00d6\u011frenciler yaz okulunda ba\u015far\u0131s\u0131z olduklar\u0131 "
        "dersleri g\u00fcz d\u00f6neminde \u00fccretsiz tekrar alabilir mi"
    )

    answer, _, _, _ = rag_llm.ask_with_clarification(
        question=question,
        clarification=None,
        history=[],
        top_k=12,
        allow_generic_rewrite=True,
    )

    assert rag_llm.is_no_answer_text(answer)
    assert llm_called is False

def test_foreign_student_application_context_contains_full_document_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages = {}

    def fake_chat(**kwargs):
        captured_messages["messages"] = kwargs["messages"]
        return {
            "message": {
                "content": "__LLM_GENERATION_CALLED__",
            }
        }

    monkeypatch.setattr(rag_llm.ollama, "chat", fake_chat)

    question = (
        "Yabanc\u0131 uyruklu \u00f6\u011frenci "
        "ba\u015fvurusunda hangi belgeler istenir?"
    )

    answer, _, _, _ = rag_llm.ask_with_clarification(
        question=question,
        clarification=None,
        history=[],
        top_k=12,
        allow_generic_rewrite=True,
    )

    assert answer == "__LLM_GENERATION_CALLED__"

    prompt_text = captured_messages["messages"][-1]["content"].lower()

    assert "ales" in prompt_text
    assert "pasaport" in prompt_text
    assert "yabancı dil" in prompt_text

def test_llm_generation_uses_reproducible_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return {
            "message": {
                "content": "__LLM_GENERATION_CALLED__",
            }
        }

    monkeypatch.setattr(rag_llm.ollama, "chat", fake_chat)

    rag_llm.ask_with_clarification(
        question="4 yıllık lisans programı için toplam AKTS kaçtır?",
        clarification=None,
        history=[],
        top_k=12,
        allow_generic_rewrite=True,
    )

    assert captured["options"]["temperature"] == 0.0
    assert captured["options"]["seed"] == 42

@pytest.mark.parametrize(
    ("question", "expected_text"),
    [
        (
            "AKTS kısaltması ne anlama gelir?",
            "Avrupa Kredi Transfer Sistemini",
        ),
        (
            "Senato nedir?",
            "Senato: Kahramanmaraş Sütçü İmam Üniversitesi Senatosunu",
        ),
    ],
)
def test_definition_questions_include_explicit_definition_context(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
    expected_text: str,
) -> None:
    captured = {}

    def fake_chat(**kwargs):
        captured["messages"] = kwargs["messages"]
        return {
            "message": {
                "content": "__LLM_GENERATION_CALLED__",
            }
        }

    monkeypatch.setattr(rag_llm.ollama, "chat", fake_chat)

    answer, _, _, _ = rag_llm.ask_with_clarification(
        question=question,
        clarification=None,
        history=[],
        top_k=12,
        allow_generic_rewrite=True,
    )

    assert answer == "__LLM_GENERATION_CALLED__"

    prompt = captured["messages"][-1]["content"]

    assert expected_text in prompt

def test_conditional_numeric_question_focuses_best_evidence_passage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_chat(**kwargs):
        captured["messages"] = kwargs["messages"]
        return {
            "message": {
                "content": "__LLM_GENERATION_CALLED__",
            }
        }

    monkeypatch.setattr(rag_llm.ollama, "chat", fake_chat)

    question = (
        "Ek sınavlara girmeden en fazla beş başarısız dersi kalan "
        "öğrenciye başarısız ders sayısını bire düşürmesi için "
        "kaç yarıyıl verilir?"
    )

    answer, _, _, _ = rag_llm.ask_with_clarification(
        question=question,
        clarification=None,
        history=[],
        top_k=12,
        allow_generic_rewrite=True,
    )

    assert answer == "__LLM_GENERATION_CALLED__"

    prompt = rag_llm._normalize_text(
        captured["messages"][-1]["content"]
    )

    assert "aralıksız 4 yarıyıl" in prompt
    assert "bu süre sonunda" in prompt
    assert "daha fazla başarısız dersi" in prompt

    assert (
        "ek sınavlar sonucu verilen ek süre 3 yarıyıl 4 yarıyıl"
        not in prompt
    )

    assert (
        "başarısız olduğu bir 1 ders"
        not in prompt
    )

    assert (
    "sorunun farklı ifadelerle aynı koşulu anlatması "
    "tek başına ret nedeni değildir"
    in prompt
)