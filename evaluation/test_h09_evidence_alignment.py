import pytest

from backend import rag_llm


pytestmark = [
    pytest.mark.llm_eval,
    pytest.mark.slow,
]


def test_h09_answer_uses_explicit_second_internship_rule():
    rag_llm.initialize_rag()

    question = (
        "Birinci stajını başarı ile bitiremeyen öğrenci "
        "ikinci stajını yapabilir mi?"
    )

    # Önce retrieval'ın doğru hükmü gerçekten getirdiğini koruyoruz.
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

    assert (
        "birinci stajını başarı ile bitiremeyen öğrenciler "
        "ikinci stajını yapamazlar"
        in normalized_context
    )

    # Sonra gerçek LLM cevabını kontrol ediyoruz.
    answer, sources, duration, retrieved = (
        rag_llm.ask_with_clarification(question)
    )

    normalized_answer = rag_llm._normalize_text(answer)

    # Ana karar bozulursa XFAIL değil, gerçek FAIL olmalı.
    assert (
        normalized_answer.startswith("hayır")
        or "ikinci stajını yapamaz" in normalized_answer
    )

    # Bilinen limitation yalnızca evidence-alignment kısmında.
    evidence_is_aligned = (
        "birinci staj" in normalized_answer
        and "ikinci staj" in normalized_answer
        and "yapamaz" in normalized_answer
    )

    if not evidence_is_aligned:
        pytest.xfail(
            "Known limitation: H09 ana karar doğru olsa da "
            "LLM bazen Madde 15/2 yerine aynı chunk içindeki "
            "alakasız Madde 15/1 cümlesini gerekçe olarak kullanıyor."
        )