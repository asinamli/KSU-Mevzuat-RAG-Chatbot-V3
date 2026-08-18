from backend import rag_llm


def test_turkish_lexical_normalizer_matches_common_inflections():
    pairs = [
        ("mazeretinin", "mazeret"),
        ("sayılması", "sayılır"),
        ("kurulun", "kurulunca"),
        ("kabulü", "kabul"),
    ]

    for left, right in pairs:
        assert (
            rag_llm._normalize_turkish_lexical_token(left)
            == rag_llm._normalize_turkish_lexical_token(right)
        )


def test_turkish_lexical_normalizer_does_not_merge_unrelated_words():
    pairs = [
        ("sınav", "sınır"),
        ("kabul", "kabak"),
        ("ders", "dergi"),
    ]

    for left, right in pairs:
        assert (
            rag_llm._normalize_turkish_lexical_token(left)
            != rag_llm._normalize_turkish_lexical_token(right)
        )