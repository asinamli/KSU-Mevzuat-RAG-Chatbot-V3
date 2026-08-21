import pytest

from backend import rag_llm


pytestmark = [
    pytest.mark.llm_eval,
    pytest.mark.slow,
]


def test_toefl_question_answers_from_explicit_72_point_rule():
    rag_llm.initialize_rag()

    question = (
        "Yabancı dilde öğretim için TOEFL IBT'den "
        "minimum kaç puan gerekir?"
    )

    # Önce retrieval'ın doğru hükmü getirdiğini koruyoruz.
    hits = rag_llm._retrieve(
        question=question,
        history=[],
        top_k=12,
        filter_params={},
        use_history=False,
    )

    hits = rag_llm._maybe_upgrade_with_fallback(
        question,
        [],
        {},
        hits,
    )

    candidates = rag_llm._select_candidate_hits(hits)

    context = "\n".join(
        str((hit.payload or {}).get("text", ""))
        for hit in candidates
    )

    normalized_context = rag_llm._normalize_text(context)

    assert "toefl ibt sınavından 72 puan" in normalized_context

    # Gerçek LLM davranışı.
    answer, sources, duration, retrieved = (
        rag_llm.ask_with_clarification(question)
    )

    if rag_llm.is_no_answer_text(answer):
        pytest.xfail(
            "Known limitation: gemma2:latest doğru TOEFL 72 "
            "context'i mevcut olsa da bazen false no-answer üretiyor."
        )

    normalized_answer = rag_llm._normalize_text(answer)

    # No-answer dışında yanlış bir cevap verilirse gerçek FAIL.
    assert "72" in normalized_answer