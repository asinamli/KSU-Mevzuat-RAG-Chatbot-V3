from __future__ import annotations

import pytest

import rag_llm


pytestmark = pytest.mark.integration


RETRIEVAL_CASES = [
    pytest.param(
        "Yandal programına hangi yarıyıllarda başvurulabilir?",
        (
            "KSÜ Çift Anadal Yandal Yönergesi "
            "9.8.2017 son_17_2012231113033925_"
            "2405031056187469_2405221138337521.docx"
        ),
        ("en erken üçüncü", "en geç altıncı"),
        3,
        id="yandal-basvuru-yariyili",
    ),
    pytest.param(
        (
            "Kurumlar arası yatay geçiş için "
            "genel not ortalaması şartı nedir?"
        ),
        (
            "KSÜ KURUM İÇİ VE KURUMLAR ARASI YATAY GEÇİŞ, "
            "DERS MUAFİYET VE İNTİBAK YÖNERGESİ "
            "12.06.2017 (Son)_24070.pdf"
        ),
        ("2.30", "60"),
        3,
        id="yatay-gecis-not-ortalamasi",
    ),
    pytest.param(
        "Teorik derslere devam mecburiyeti yüzde kaçtır?",
        (
            "KSÜ LİSANS EĞİTİM ÖĞRETİM YÖNETMELİK SON "
            "(23 Haziran 2019)_1906271417371007_"
            "2110130857557610_21101810.docx"
        ),
        ("%70", "%80"),
        3,
        id="teorik-ders-devam-orani",
    ),
    pytest.param(
        "UME dersi hangi bölümler için zorunludur?",
        (
            "KSÜ Mühendislik ve Mimarlık Fakültesi "
            "Uygulamalı Mühendislik Eğitimi "
            "(UME) Yönergesi_2112241121266583.pdf"
        ),
        ("Tekstil Mühendisliği", "zorunludur"),
        3,
        id="ume-zorunlu-bolum",
    ),
]


@pytest.fixture(scope="module", autouse=True)
def initialize_real_rag() -> None:
    rag_llm.initialize_rag()


@pytest.mark.parametrize(
    (
        "question",
        "expected_source",
        "expected_text_parts",
        "maximum_rank",
    ),
    RETRIEVAL_CASES,
)
def test_retrieval_returns_answer_supporting_chunk(
    question: str,
    expected_source: str,
    expected_text_parts: tuple[str, ...],
    maximum_rank: int,
) -> None:
    # Act
    hits = rag_llm._retrieve(
        question=question,
        history=[],
        top_k=10,
        filter_params=None,
        use_history=False,
    )

    # Assert
    assert hits

    matching_ranks: list[int] = []

    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload
        source = str(payload.get("source", ""))
        text = str(payload.get("text", "")).casefold()

        contains_expected_answer = all(
            part.casefold() in text
            for part in expected_text_parts
        )

        if (
            source == expected_source
            and contains_expected_answer
        ):
            matching_ranks.append(rank)

    assert matching_ranks, (
        "Beklenen cevabı taşıyan chunk ilk 10 sonuçta bulunamadı. "
        f"Soru: {question}"
    )

    assert min(matching_ranks) <= maximum_rank, (
        "Beklenen cevabı taşıyan chunk yeterince üst sırada değil. "
        f"Soru: {question}, "
        f"sıralar: {matching_ranks}, "
        f"beklenen en yüksek sıra: {maximum_rank}"
    )