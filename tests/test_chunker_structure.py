from __future__ import annotations

from chunker import ChunkConfig, chunk_mevzuat, extract_facets


TEST_CONFIG = ChunkConfig(
    chunk_size=5000,
    chunk_overlap=0,
    min_chunk_size=1,
)


def test_docx_table_content_is_preserved_and_markers_are_removed() -> None:
    document = {
        "source": "ornek_ders_plani.docx",
        "ext": ".docx",
        "text": (
            "MADDE 1 Ders planı\n"
            "[DOCX_TABLE]\n"
            "Ders | AKTS\n"
            "Yapay Zeka | 5 AKTS\n"
            "Veri Tabanı | 6 AKTS\n"
            "[/DOCX_TABLE]"
        ),
    }

    chunks = chunk_mevzuat(
        [document],
        cfg=TEST_CONFIG,
    )

    assert len(chunks) == 1

    chunk = chunks[0]

    assert "[DOCX_TABLE]" not in chunk["text"]
    assert "[/DOCX_TABLE]" not in chunk["text"]

    assert "Ders | AKTS" in chunk["text"]
    assert "Yapay Zeka | 5 AKTS" in chunk["text"]
    assert "Veri Tabanı | 6 AKTS" in chunk["text"]

    assert chunk["madde_no"] == 1
    assert chunk["madde_kind"] == "madde"

    assert chunk["contains_akts"] is True
    assert chunk["akts_values"] == [5, 6]


def test_docx_table_stays_with_the_correct_article() -> None:
    document = {
        "source": "ornek_yonetmelik.docx",
        "ext": ".docx",
        "text": (
            "MADDE 1 Ders yükü\n"
            "Öğrencinin alabileceği dersler aşağıdaki tabloda gösterilir.\n"
            "[DOCX_TABLE]\n"
            "Ders | AKTS\n"
            "Programlama | 6 AKTS\n"
            "[/DOCX_TABLE]\n\n"
            "MADDE 2 Devam koşulu\n"
            "Teorik derslere devam zorunludur."
        ),
    }

    chunks = chunk_mevzuat(
        [document],
        cfg=TEST_CONFIG,
    )

    assert len(chunks) == 2

    first_chunk = chunks[0]
    second_chunk = chunks[1]

    assert first_chunk["madde_no"] == 1
    assert "Programlama | 6 AKTS" in first_chunk["text"]

    assert second_chunk["madde_no"] == 2
    assert "Teorik derslere devam zorunludur." in second_chunk["text"]

    assert "Programlama | 6 AKTS" not in second_chunk["text"]


def test_pdf_page_markers_create_page_metadata() -> None:
    document = {
        "source": "ornek_yonetmelik.pdf",
        "ext": ".pdf",
        "pdf_pages": 3,
        "pdf_low_text": False,
        "text": (
            "MADDE 7 Devam zorunluluğu\n"
            "[PDF_PAGE 2]\n"
            "Teorik derslere en az yüzde 70 devam zorunludur.\n"
            "[PDF_PAGE 3]\n"
            "Uygulamalı derslere en az yüzde 80 devam zorunludur."
        ),
    }

    chunks = chunk_mevzuat(
        [document],
        cfg=TEST_CONFIG,
    )

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk["madde_no"] == 7
    assert chunk["page_start"] == 2
    assert chunk["page_end"] == 3

    assert "[PDF_PAGE 2]" not in chunk["text"]
    assert "[PDF_PAGE 3]" not in chunk["text"]

    assert "yüzde 70" in chunk["text"]
    assert "yüzde 80" in chunk["text"]

    assert chunk["ext"] == ".pdf"
    assert chunk["pdf_pages"] == 3
    assert chunk["pdf_low_text"] is False


def test_bir_yariyilda_expression_maps_to_normal_term_scope() -> None:
    facets = extract_facets(
        "Bir yarıyılda en fazla 46 AKTS kredisi ders alınabilir."
    )

    assert facets["term_scope"] == "normal_donem"


def test_yaz_ogretiminde_with_normal_term_signal_is_mixed() -> None:
    facets = extract_facets(
        "Yaz öğretiminde bir yarıyılda ders alınabilir."
    )

    assert facets["term_scope"] == "mixed"


def test_orgun_ogretimde_maps_to_orgun_teaching_mode() -> None:
    facets = extract_facets(
        "Örgün öğretimde bir dersin başarı hesabı yapılır."
    )

    assert facets["teaching_mode"] == "orgun"


def test_uzaktan_ogretimde_maps_to_uzaktan_teaching_mode() -> None:
    facets = extract_facets(
        "Uzaktan öğretimde dersler yürütülür."
    )

    assert facets["teaching_mode"] == "uzaktan"


def test_same_text_with_orgun_and_uzaktan_is_mixed() -> None:
    facets = extract_facets(
        "Uzaktan öğretim yoluyla örgün öğretimde ders verilebilir."
    )

    assert facets["teaching_mode"] == "mixed"