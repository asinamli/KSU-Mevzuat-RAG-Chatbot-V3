import pytest

from file_reader import _clean_extracted_text


@pytest.mark.parametrize(
    ("raw_text", "expected_text"),
    [
        (
            "  KSÜ   Mevzuat\r\nDanışmanı  ",
            "KSÜ Mevzuat\nDanışmanı",
        ),
        (
            "öğren-\nci",
            "öğrenci",
        ),
        (
            "yönetmelik\u00ad",
            "yönetmelik",
        ),
        (
            "birinci satır\n\n\n\nikinci satır",
            "birinci satır\n\nikinci satır",
        ),
        (
            "",
            "",
        ),
    ],
)
def test_clean_extracted_text(
    raw_text: str,
    expected_text: str,
) -> None:
    actual_text = _clean_extracted_text(raw_text)

    assert actual_text == expected_text
