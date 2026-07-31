from __future__ import annotations

from pathlib import Path

import pytest
from docx import Document

from file_reader import ReadConfig, read_all_documents


pytestmark = pytest.mark.integration

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

EXPECTED_DOCUMENT_COUNT = 22
EXPECTED_PDF_COUNT = 8
EXPECTED_DOCX_COUNT = 14

COMMON_DOCUMENT_KEYS = {
    "source",
    "source_url",
    "path",
    "ext",
    "size_bytes",
    "mtime",
    "text",
}


def _list_expected_input_files(
    files_dir: Path,
) -> list[Path]:
    return sorted(
        (
            path
            for path in files_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and not path.name.startswith("~$")
        ),
        key=lambda path: str(path).lower(),
    )


def test_real_documents_are_read_without_silent_skips(
    real_documents_dir: Path,
) -> None:
    # Arrange:
    # Diskte bulunması gereken gerçek girdi kümesini
    # read_all_documents'tan bağımsız olarak say.
    input_files = _list_expected_input_files(
        real_documents_dir,
    )

    pdf_files = [
        path
        for path in input_files
        if path.suffix.lower() == ".pdf"
    ]

    docx_files = [
        path
        for path in input_files
        if path.suffix.lower() == ".docx"
    ]

    expected_sources = {
        path.relative_to(real_documents_dir).as_posix()
        for path in input_files
    }

    assert len(input_files) == EXPECTED_DOCUMENT_COUNT
    assert len(pdf_files) == EXPECTED_PDF_COUNT
    assert len(docx_files) == EXPECTED_DOCX_COUNT

    # Act:
    # Gerçek PDF ve DOCX okuyucuları ile gerçek klasörü oku.
    documents = read_all_documents(
        files_dir=real_documents_dir,
        cfg=ReadConfig(),
    )

    # Assert:
    # Hiçbir belge sessizce atlanmamalı veya fazladan eklenmemeli.
    assert len(documents) == EXPECTED_DOCUMENT_COUNT

    actual_sources = [
        str(document["source"])
        for document in documents
    ]

    assert len(actual_sources) == len(set(actual_sources))
    assert set(actual_sources) == expected_sources

    low_text_pdf_sources: list[str] = []

    for document in documents:
        assert isinstance(document, dict)
        assert COMMON_DOCUMENT_KEYS.issubset(
            document.keys()
        )

        source = document["source"]
        text = document["text"]
        extension = document["ext"]
        stored_path = document["path"]

        assert isinstance(source, str)
        assert source.strip()

        assert isinstance(text, str)
        assert text.strip()

        assert extension in SUPPORTED_EXTENSIONS

        assert isinstance(stored_path, str)
        assert stored_path.strip()

        assert isinstance(document["size_bytes"], int)
        assert document["size_bytes"] > 0

        assert isinstance(document["mtime"], int)

        expected_path = (
            real_documents_dir / Path(source)
        ).resolve()

        assert Path(stored_path).resolve() == expected_path

        if extension == ".pdf":
            # Önce metadata yapısını doğrula.
            assert "pdf_pages" in document
            assert isinstance(
                document["pdf_pages"],
                int,
            )
            assert document["pdf_pages"] > 0

            assert "pdf_low_text" in document
            assert isinstance(
                document["pdf_low_text"],
                bool,
            )

            assert "pdf_error" not in document

            # Yapı doğrulandıktan sonra gerçek değeri değerlendir.
            if document["pdf_low_text"]:
                low_text_pdf_sources.append(source)

    assert low_text_pdf_sources == []


def test_temporary_word_lock_files_are_ignored(
    tmp_path: Path,
) -> None:
    # Arrange:
    # Normal DOCX ve Word'ün oluşturabileceği ~$ kilit dosyası.
    normal_path = tmp_path / "document.docx"
    temporary_path = tmp_path / "~$document.docx"

    for path in (normal_path, temporary_path):
        document = Document()
        document.add_paragraph(
            "KSÜ mevzuat test belgesi"
        )
        document.save(path)

    # Act
    documents = read_all_documents(
        files_dir=tmp_path,
        cfg=ReadConfig(recursive=False),
    )

    # Assert:
    # Yalnızca gerçek belge ingestion girdisi olmalı.
    assert [
        document["source"]
        for document in documents
    ] == ["document.docx"]