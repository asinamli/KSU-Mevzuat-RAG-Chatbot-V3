from backend import rag_llm


def _hit(score: float, text: str) -> rag_llm.RankedHit:
    return rag_llm.RankedHit(
        payload={"text": text},
        score=score,
        dense_score=score,
        lexical_score=0.0,
        rerank_score=score,
    )


def test_candidate_selection_keeps_near_tie_after_primary_four():
    hits = [
        _hit(0.700, "candidate-a"),
        _hit(0.680, "candidate-b"),
        _hit(0.670, "candidate-c"),
        _hit(0.660, "candidate-d"),
        _hit(0.655, "candidate-e-near-tie"),
        _hit(0.620, "candidate-f-not-near-tie"),
    ]

    selected = rag_llm._select_candidate_hits(hits)

    selected_texts = [
        hit.payload["text"]
        for hit in selected
    ]

    assert "candidate-e-near-tie" in selected_texts
    assert "candidate-f-not-near-tie" not in selected_texts
    assert len(selected) == 5