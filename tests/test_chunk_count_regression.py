from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from chunker import ChunkConfig, chunk_mevzuat
from file_reader import ReadConfig, read_all_documents


pytestmark = pytest.mark.integration

EXPECTED_DOCUMENT_COUNT = 22
EXPECTED_TOTAL_CHUNKS = 748

EXPECTED_CHUNKS_BY_SOURCE = {
    "Ek Sınav Uygulama Esasları_2508121355179287.pdf": 10,
    "generate.pdf": 33,
    "ksu_yabancı_dil.docx": 31,
    "KSÜ Akademik Danışmanlık Yönergesi son şekli_1507131457120770.docx": 6,
    "KSÜ FORMASYON YÖNERGESİ-2017.docx": 30,
    "KSÜ KURUM İÇİ VE KURUMLAR ARASI YATAY GEÇİŞ, DERS MUAFİYET VE İNTİBAK YÖNERGESİ 12.06.2017 (Son)_24070.pdf": 41,
    "KSÜ Lisans bitirme Tezi yönerge_1702011029595839.docx": 34,
    "KSÜ Lisanüstü Yabancı Öğrenci Yönergesi SON_1711281619022410.docx": 25,
    "KSÜ LİSANS EĞİTİM ÖĞRETİM YÖNETMELİK SON (23 Haziran 2019)_1906271417371007_2110130857557610_21101810.docx": 123,
    "KSÜ Mühendislik ve Mimarlık Fakültesi Uygulamalı Mühendislik Eğitimi (UME) Yönergesi_2112241121266583.pdf": 21,
    "KSÜ Staj Yönergesi 9.8.2017 son_1812101440091197.docx": 17,
    "KSÜ UZAKTAN ÖĞRETİM YÖNERGESİ_2212071543444445.pdf": 33,
    "KSÜ Yaz Öğretimi Yönergesi_pdf_2406060945047358.pdf": 19,
    "KSÜ Çift Anadal Yandal Yönergesi 9.8.2017 son_17_2012231113033925_2405031056187469_2405221138337521.docx": 26,
    "KSÜ ÖNLİSANS-LİSANS ULUSLARARASI ÖĞRENCİ BAŞVURU KABUL VE KAYIT YÖNERGESİ_2405221321054057.docx": 21,
    "KSÜ Özel öğrenci yönergesi_2309211343292066.pdf": 13,
    "KSÜ Öğrenci Konseyi Yönergesi (son) _2210311238289464.pdf": 35,
    "KSÜ Öğrenci topluluk YÖNERGESİ SON_1507131501251906.docx": 41,
    "KSÜ ÖĞRENCİ DİSİPLİN YÖNETMELİĞİ_2401181610214594.docx": 33,
    "KSÜ-ÖNLİSANS-LİSANS YÖNETMELİĞİ_1507131506129631.docx": 71,
    "Yönerge_son_hali_2412261417017378.docx": 50,
    "YÜKSEKÖĞRETİM KURUMLARI UYGULAMALI EĞİTİMLER ÇERÇEVE YÖNETMELİĞİ_2111231107088196.docx": 35,
}


def test_real_documents_keep_expected_chunk_counts(
    real_documents_dir: Path,
) -> None:
    # Arrange
    documents = read_all_documents(
        files_dir=real_documents_dir,
        cfg=ReadConfig(),
    )

    # Act
    chunks = chunk_mevzuat(
        documents=documents,
        cfg=ChunkConfig(),
    )

    actual_chunks_by_source = Counter(
        str(chunk["source"])
        for chunk in chunks
    )

    # Assert
    assert len(documents) == EXPECTED_DOCUMENT_COUNT

    assert dict(actual_chunks_by_source) == EXPECTED_CHUNKS_BY_SOURCE

    assert len(chunks) == EXPECTED_TOTAL_CHUNKS