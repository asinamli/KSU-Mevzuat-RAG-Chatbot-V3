from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PDF_PAGE_RE = re.compile(r"\[PDF_PAGE\s+(?P<page>\d+)\]", re.IGNORECASE)
DOCX_TABLE_OPEN_RE = re.compile(r"\[DOCX_TABLE\]", re.IGNORECASE)
DOCX_TABLE_CLOSE_RE = re.compile(r"\[/DOCX_TABLE\]", re.IGNORECASE)

ARTICLE_HEADER_RE = re.compile(
    r"(?im)^(?P<header>(?:(?P<kind>GEÇİCİ|GECICI|EK)\s+)?MADDE\s+(?P<no>\d+)\b[^\n]*)"
)

TERM_SCOPE_RULES: List[Tuple[str, re.Pattern]] = [
    (
        "yaz_okulu",
        re.compile(
            r"\byaz\s*(okulu|öğretimi(?:nde)?|ogretimi(?:nde)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "normal_donem",
        re.compile(
            r"\b("
            r"normal\s*d[öo]nem|"
            r"g[üu]z\s*d[öo]nemi|"
            r"bahar\s*d[öo]nemi|"
            r"yar[ıi]y[ıi]l|"
            r"bir\s+yar[ıi]y[ıi]lda"
            r")\b",
            re.IGNORECASE,
        ),
    ),
]

PROGRAM_LEVEL_RULES: List[Tuple[str, re.Pattern]] = [
    ("onlisans", re.compile(r"\bön\s*lisans\b|\bon\s*lisans\b", re.IGNORECASE)),
    ("lisansustu", re.compile(r"\blisans\s*[-]?\s*üstü\b|\blisansüstü\b|\blisansustu\b", re.IGNORECASE)),
    ("yuksek_lisans", re.compile(r"\b(yüksek|yuksek)\s*lisans\b", re.IGNORECASE)),
    ("doktora", re.compile(r"\b(doktora|ph\.?\s*d)\b", re.IGNORECASE)),
    ("lisans", re.compile(r"\blisans\b(?![\s-]*üstü)", re.IGNORECASE)),
]

# Buradaki kritik düzeltme:
# 'kredi' tek başına artık ders yükü sinyali değil.
# Çünkü 'kredi transferi' gibi ifadeler çift anadal / yatay geçiş metinlerini yanlış etiketliyordu.
TOPIC_RULES: List[Tuple[str, re.Pattern]] = [
    ("akts", re.compile(r"\bAKTS\b", re.IGNORECASE)),
    (
        "ders_yuku",
        re.compile(
            r"\b("
            r"ders\s*yükü|ders\s*yuku|ders\s*alma|"
            r"bir\s*yarıyılda\s*alınabilecek|bir\s*yariyilda\s*alinabilecek|"
            r"yarıyılda\s*alınabilecek|yariyilda\s*alinabilecek|"
            r"toplam\s*kredi|alınabilecek\s*kredi|alinabilecek\s*kredi|"
            r"en\s*fazla\s*\d+\s*(AKTS|kredi)"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    ("mezuniyet", re.compile(r"\bmezun(iyet|olmak|olabil)?\b", re.IGNORECASE)),
    ("sinav", re.compile(r"\b(sınav|vize|final|bütünleme|butunleme|mazeret)\b", re.IGNORECASE)),
    ("devamsizlik", re.compile(r"\b(devam|devamsızlık|devamsizlik|yoklama)\b", re.IGNORECASE)),
    (
        "kayit",
        re.compile(r"\b(kayıt|kayit|ders\s*kayıt|ders\s*kayit|kayıt\s*silme|kayit\s*silme)\b", re.IGNORECASE),
    ),
    ("staj", re.compile(r"\bstaj\b", re.IGNORECASE)),
]

STUDENT_STATUS_RULES: List[Tuple[str, re.Pattern]] = [
    ("cift_anadal_yandal", re.compile(r"\b(çift\s*anadal|cift\s*anadal|yandal|yan\s*dal)\b", re.IGNORECASE)),
    ("ozel_ogrenci", re.compile(r"\b(özel\s*öğrenci|ozel\s*ogrenci)\b", re.IGNORECASE)),
    (
        "uluslararasi_ogrenci",
        re.compile(
            r"\b(uluslararası\s*öğrenci|uluslararasi\s*ogrenci|yabancı\s*uyruklu|yabanci\s*uyruklu|yabancı\s*öğrenci|yabanci\s*ogrenci)\b",
            re.IGNORECASE,
        ),
    ),
]

TEACHING_MODE_RULES: List[Tuple[str, re.Pattern]] = [
    (
        "uzaktan",
        re.compile(
            r"\b(uzaktan\s*öğretim(?:de)?|uzaktan\s*ogretim(?:de)?|çevrimiçi|cevrimici|eş\s*zamanlı|es\s*zamanli|eş\s*zamansız|es\s*zamansiz|harmanlanmış|harmanlanmis|karma)\b",
            re.IGNORECASE,
        ),
    ),
    ("orgun", re.compile(r"\b(örgün\s*öğretim(?:de)?|orgun\s*ogretim(?:de)?|yüz\s*yüze|yuz\s*yuze)\b", re.IGNORECASE)),
]

# Burada artık bazı aşırı geniş domain'ler parçalandı.
# Amaç: birbirine yakın mevzuat kümelerini aynı torbaya atmayıp daha seçici etiketlemek.
TOPIC_DOMAIN_RULES: List[Tuple[str, re.Pattern]] = [
    ("yaz_ogretimi", re.compile(r"\b(yaz\s*öğretimi|yaz\s*ogretimi|yaz\s*okulu)\b", re.IGNORECASE)),
    ("uzaktan_ogretim", re.compile(r"\b(uzaktan\s*öğretim|uzaktan\s*ogretim|çevrimiçi|cevrimici|harmanlanmış|harmanlanmis|karma)\b", re.IGNORECASE)),
    ("ozel_ogrenci", re.compile(r"\b(özel\s*öğrenci|ozel\s*ogrenci)\b", re.IGNORECASE)),
    (
        "uluslararasi_ogrenci",
        re.compile(r"\b(uluslararası\s*öğrenci|uluslararasi\s*ogrenci|yabancı\s*uyruklu|yabanci\s*uyruklu)\b", re.IGNORECASE),
    ),
    (
        "lisansustu_yabanci_ogrenci",
        re.compile(
            r"\b(lisansüstü\s*yabancı\s*öğrenci|lisansustu\s*yabanci\s*ogrenci|"
            r"yabancı\s*uyruklu\s*lisansüstü|yabanci\s*uyruklu\s*lisansustu|"
            r"enstitü\s*başvuru|enstitu\s*basvuru|lisansüstü\s*başvuru|lisansustu\s*basvuru)\b",
            re.IGNORECASE,
        ),
    ),
    ("cift_anadal_yandal", re.compile(r"\b(çift\s*anadal|cift\s*anadal|yandal|yan\s*dal)\b", re.IGNORECASE)),
    ("yatay_gecis_intibak", re.compile(r"\b(yatay\s*geçiş|yatay\s*gecis|muafiyet|intibak)\b", re.IGNORECASE)),
    (
        "ume",
        re.compile(r"\b(ume|uygulamalı\s*mühendislik\s*eğitimi|uygulamali\s*muhendislik\s*egitimi)\b", re.IGNORECASE),
    ),
    (
        "uygulamali_egitim",
        re.compile(r"\b(uygulamalı\s*eğitim|uygulamali\s*egitim|uygulamalı\s*ders|uygulamali\s*ders|işletmede\s*mesleki\s*eğitim|isletmede\s*mesleki\s*egitim|iş\s*yeri\s*uygulaması|is\s*yeri\s*uygulamasi)\b", re.IGNORECASE),
    ),
    ("staj", re.compile(r"\bstaj\b", re.IGNORECASE)),
    ("disiplin", re.compile(r"\b(disiplin|uyarma|kınama|kinama|uzaklaştırma|uzaklastirma)\b", re.IGNORECASE)),
    ("akademik_danismanlik", re.compile(r"\b(akademik\s*danışman|akademik\s*danisman)\b", re.IGNORECASE)),
    (
        "yabanci_dil_hazirlik",
        re.compile(r"\b(hazırlık\s*programı|hazirlik\s*programi|hazırlık\s*sınıfı|hazirlik\s*sinifi|yeterlilik\s*sınavı|yeterlilik\s*sinavi|seviye\s*tespit|muafiyet|yabancı\s*dil\s*hazırlık|yabanci\s*dil\s*hazirlik)\b", re.IGNORECASE),
    ),
    (
        "yabanci_dille_ogretim",
        re.compile(r"\b(yabancı\s*dille\s*öğretim|yabanci\s*dille\s*ogretim|öğretim\s*dili\s*yabancı\s*dil|ogretim\s*dili\s*yabanci\s*dil|tamamen\s*veya\s*en\s*az\s*%\s*30\s*yabancı|tamamen\s*veya\s*en\s*az\s*%\s*30\s*yabanci)\b", re.IGNORECASE),
    ),
    ("ek_sinav", re.compile(r"\b(ek\s*sınav|ek\s*sinav|azami\s*süre|azami\s*sure)\b", re.IGNORECASE)),
    (
        "ders_yuku",
        re.compile(
            r"\b("
            r"AKTS|ders\s*yükü|ders\s*yuku|ders\s*alma|"
            r"bir\s*yarıyılda\s*alınabilecek|bir\s*yariyilda\s*alinabilecek|"
            r"yarıyılda\s*alınabilecek|yariyilda\s*alinabilecek|"
            r"toplam\s*kredi|alınabilecek\s*kredi|alinabilecek\s*kredi"
            r")\b",
            re.IGNORECASE,
        ),
    ),
]

YABANCI_DIL_HAZIRLIK_HINT_RE = re.compile(
    r"\b(hazırlık\s*programı|hazirlik\s*programi|hazırlık\s*sınıfı|hazirlik\s*sinifi|yeterlilik\s*sınavı|yeterlilik\s*sinavi|seviye\s*tespit|muafiyet)\b",
    re.IGNORECASE,
)
YABANCI_DILLE_OGRETIM_HINT_RE = re.compile(
    r"\b(yabancı\s*dille\s*öğretim|yabanci\s*dille\s*ogretim|öğretim\s*dili\s*yabancı\s*dil|ogretim\s*dili\s*yabanci\s*dil|tamamen\s*veya\s*en\s*az\s*%\s*30\s*yabancı|tamamen\s*veya\s*en\s*az\s*%\s*30\s*yabanci)\b",
    re.IGNORECASE,
)
LISANSUSTU_YABANCI_OGRENCI_HINT_RE = re.compile(
    r"\b(lisansüstü\s*yabancı\s*öğrenci|lisansustu\s*yabanci\s*ogrenci|yabancı\s*uyruklu\s*lisansüstü|yabanci\s*uyruklu\s*lisansustu|enstitü|enstitu)\b",
    re.IGNORECASE,
)
UME_HINT_RE = re.compile(r"\b(ume|uygulamalı\s*mühendislik\s*eğitimi|uygulamali\s*muhendislik\s*egitimi)\b", re.IGNORECASE)
UYGULAMALI_EGITIM_HINT_RE = re.compile(r"\b(uygulamalı\s*eğitim|uygulamali\s*egitim|uygulamalı\s*ders|uygulamali\s*ders|işletmede\s*mesleki\s*eğitim|isletmede\s*mesleki\s*egitim|iş\s*yeri\s*uygulaması|is\s*yeri\s*uygulamasi)\b", re.IGNORECASE)
STAJ_HINT_RE = re.compile(r"\bstaj\b", re.IGNORECASE)

AKTS_NUM_RE = re.compile(r"(\d{1,3})\s*AKTS", re.IGNORECASE)
TEZLI_RE = re.compile(r"\btezli\b", re.IGNORECASE)
TEZSIZ_RE = re.compile(r"\btezsiz\b", re.IGNORECASE)

_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")
_SOFT_HYPHEN = "\u00ad"
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)", flags=re.UNICODE)


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = 1100
    chunk_overlap: int = 220
    separators: Tuple[str, ...] = ("\n\n", "\n", ". ", "; ", ": ", ", ", " ", "")
    min_chunk_size: int = 200
    strip_pdf_page_markers: bool = True
    strip_docx_table_markers: bool = True
    skip_low_text_pdfs: bool = False
    passthrough_keys: Tuple[str, ...] = ("source_url", "ext", "mtime", "size_bytes", "pdf_pages", "pdf_low_text", "sha256", "path")


def normalize_text_keep_structure(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(_SOFT_HYPHEN, "")
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MANY_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def _extract_page_range(text: str) -> Tuple[Optional[int], Optional[int]]:
    pages = [int(m.group("page")) for m in PDF_PAGE_RE.finditer(text or "")]
    if not pages:
        return None, None
    return min(pages), max(pages)


def _strip_markers(text: str, cfg: ChunkConfig) -> str:
    out = text or ""
    if cfg.strip_pdf_page_markers:
        out = PDF_PAGE_RE.sub("\n", out)
    if cfg.strip_docx_table_markers:
        out = DOCX_TABLE_OPEN_RE.sub("", out)
        out = DOCX_TABLE_CLOSE_RE.sub("", out)
    return normalize_text_keep_structure(out)


def _split_with_sep(text: str, sep: str) -> List[str]:
    if sep == "":
        return [text]
    parts = text.split(sep)
    if len(parts) == 1:
        return [text]
    out: List[str] = []
    for p in parts[:-1]:
        out.append((p + sep) if p else sep)
    if parts[-1]:
        out.append(parts[-1])
    return out


def _recursive_split(text: str, cfg: ChunkConfig, length_fn: Callable[[str], int]) -> List[str]:
    if length_fn(text) <= cfg.chunk_size:
        return [text]

    def _split_level(t: str, seps: Sequence[str]) -> List[str]:
        if length_fn(t) <= cfg.chunk_size or not seps:
            return [t]
        sep = seps[0]
        splits = _split_with_sep(t, sep)
        if len(splits) == 1:
            return _split_level(t, seps[1:])
        out: List[str] = []
        for s in splits:
            s = s.strip()
            if not s:
                continue
            out.extend([s] if length_fn(s) <= cfg.chunk_size else _split_level(s, seps[1:]))
        return out

    atomic = _split_level(text, cfg.separators)

    def _take_overlap(prev: str) -> str:
        if cfg.chunk_overlap <= 0 or not prev:
            return ""
        tail = prev[-cfg.chunk_overlap:] if len(prev) > cfg.chunk_overlap else prev
        m = re.search(r"\s", tail)
        return tail[m.start():] if m else tail

    merged: List[str] = []
    cur = ""
    for piece in atomic:
        piece = piece.strip()
        if not piece:
            continue
        if not cur:
            cur = piece
            continue
        if length_fn(cur) + 1 + length_fn(piece) <= cfg.chunk_size:
            cur = (cur + "\n" + piece).strip()
        else:
            merged.append(cur)
            cur = (_take_overlap(cur) + "\n" + piece).strip()
    if cur:
        merged.append(cur)

    compact: List[str] = []
    for ch in merged:
        if compact and length_fn(ch) < cfg.min_chunk_size and length_fn(compact[-1]) + 1 + length_fn(ch) <= cfg.chunk_size:
            compact[-1] = (compact[-1] + "\n" + ch).strip()
        else:
            compact.append(ch)
    return compact


def _infer_candidates(text: str, rules: List[Tuple[str, re.Pattern]]) -> List[str]:
    found: List[str] = []
    for label, pat in rules:
        if pat.search(text or ""):
            found.append(label)
    seen = set()
    out: List[str] = []
    for x in found:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _pick_topic_domain(text: str, domain_cand: List[str]) -> str:
    if not domain_cand:
        return "general"

    text = text or ""
    found = set(domain_cand)

    if "lisansustu_yabanci_ogrenci" in found or LISANSUSTU_YABANCI_OGRENCI_HINT_RE.search(text):
        return "lisansustu_yabanci_ogrenci"

    if "ume" in found or UME_HINT_RE.search(text):
        return "ume"

    has_yd_haz = "yabanci_dil_hazirlik" in found or YABANCI_DIL_HAZIRLIK_HINT_RE.search(text)
    has_yd_ogrt = "yabanci_dille_ogretim" in found or YABANCI_DILLE_OGRETIM_HINT_RE.search(text)
    if has_yd_haz and not has_yd_ogrt:
        return "yabanci_dil_hazirlik"
    if has_yd_ogrt and not has_yd_haz:
        return "yabanci_dille_ogretim"
    if has_yd_haz and has_yd_ogrt:
        return "mixed"

    has_uyg = "uygulamali_egitim" in found or UYGULAMALI_EGITIM_HINT_RE.search(text)
    has_staj = "staj" in found or STAJ_HINT_RE.search(text)
    if has_uyg and not has_staj:
        return "uygulamali_egitim"
    if has_staj and not has_uyg:
        return "staj"
    if has_uyg and has_staj:
        return "mixed"

    return domain_cand[0] if len(domain_cand) == 1 else "mixed"


def extract_facets(text: str) -> Dict[str, Any]:
    term_cand = _infer_candidates(text, TERM_SCOPE_RULES)
    prog_cand = _infer_candidates(text, PROGRAM_LEVEL_RULES)
    topic_cand = _infer_candidates(text, TOPIC_RULES)
    student_cand = _infer_candidates(text, STUDENT_STATUS_RULES)
    mode_cand = _infer_candidates(text, TEACHING_MODE_RULES)
    domain_cand = _infer_candidates(text, TOPIC_DOMAIN_RULES)

    if "lisansustu" in prog_cand and "lisans" in prog_cand:
        prog_cand = [x for x in prog_cand if x != "lisans"]

    term_scope = term_cand[0] if len(term_cand) == 1 else ("mixed" if len(term_cand) > 1 else "unknown")
    program_level = prog_cand[0] if len(prog_cand) == 1 else ("mixed" if len(prog_cand) > 1 else "unknown")
    student_status = student_cand[0] if len(student_cand) == 1 else ("mixed" if len(student_cand) > 1 else "normal")
    teaching_mode = mode_cand[0] if len(mode_cand) == 1 else ("mixed" if len(mode_cand) > 1 else "unknown")
    topic_domain = _pick_topic_domain(text, domain_cand)

    # Kritik gürültü azaltma:
    # Çift anadal / yatay geçiş gibi domain'lerde yalnızca yan cümlede geçen kredi ifadesi yüzünden
    # topic=ders_yuku oluşmasın.
    if topic_domain in {"cift_anadal_yandal", "yatay_gecis_intibak"} and "ders_yuku" in topic_cand and "akts" not in topic_cand:
        topic_cand = [x for x in topic_cand if x != "ders_yuku"]

    if not topic_cand:
        topic_primary = "unknown"
    elif "akts" in topic_cand:
        topic_primary = "akts"
    elif len(topic_cand) == 1:
        topic_primary = topic_cand[0]
    else:
        topic_primary = "mixed"

    akts_vals = sorted({int(m.group(1)) for m in AKTS_NUM_RE.finditer(text or "")})

    return {
        "term_scope": term_scope,
        "term_scope_candidates": term_cand,
        "program_level": program_level,
        "program_level_candidates": prog_cand,
        "student_status": student_status,
        "student_status_candidates": student_cand,
        "teaching_mode": teaching_mode,
        "teaching_mode_candidates": mode_cand,
        "topic_domain": topic_domain,
        "topic": topic_primary,
        "topic_candidates": topic_cand,
        "akts_values": akts_vals,
        "contains_akts": bool(akts_vals),
        "has_tezli": bool(TEZLI_RE.search(text or "")),
        "has_tezsiz": bool(TEZSIZ_RE.search(text or "")),
    }


def _article_kind_norm(kind: Optional[str]) -> str:
    if not kind:
        return "madde"
    k = kind.strip().upper()
    if k in {"GEÇİCİ", "GECICI"}:
        return "gecici_madde"
    if k == "EK":
        return "ek_madde"
    return "madde"


def _split_into_articles(text: str) -> List[Dict[str, Any]]:
    matches = list(ARTICLE_HEADER_RE.finditer(text or ""))
    if not matches:
        return [{"madde_kind": "unknown", "madde_no": None, "madde_header": "", "span_text": text or ""}]

    spans: List[Dict[str, Any]] = []

    if matches[0].start() > 0:
        spans.append(
            {
                "madde_kind": "preamble",
                "madde_no": None,
                "madde_header": "",
                "span_text": text[: matches[0].start()],
            }
        )

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if (i + 1) < len(matches) else len(text)
        header = (match.group("header") or "").strip()
        kind = _article_kind_norm(match.group("kind"))
        no = int(match.group("no")) if match.group("no") else None

        spans.append(
            {
                "madde_kind": kind,
                "madde_no": no,
                "madde_header": header,
                "span_text": text[start:end].strip(),
            }
        )

    return spans


def _stable_uid(*parts: Any) -> str:
    raw = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def chunk_mevzuat(documents: list, cfg: ChunkConfig = ChunkConfig()) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []

    for doc in documents:
        raw_text = str(doc.get("text", "") or "")
        if not raw_text.strip():
            continue

        source = str(doc.get("source", "unknown") or "unknown")

        if cfg.skip_low_text_pdfs and doc.get("ext") == ".pdf" and doc.get("pdf_low_text"):
            logger.warning("Düşük metin PDF atlandı: %s", source)
            continue

        normalized = normalize_text_keep_structure(raw_text)
        spans = _split_into_articles(normalized)

        for span_idx, span in enumerate(spans):
            raw_span = span["span_text"]
            if not raw_span.strip():
                continue

            page_start, page_end = _extract_page_range(raw_span)
            clean_span = _strip_markers(raw_span, cfg)
            if not clean_span.strip():
                continue

            madde_header = span.get("madde_header", "") or ""
            madde_kind = span.get("madde_kind", "unknown")
            madde_no = span.get("madde_no", None)

            pieces = _recursive_split(clean_span, cfg, length_fn=len)

            for chunk_seq, piece in enumerate(pieces):
                piece = piece.strip()
                if not piece:
                    continue

                text_out = piece
                embed_text = (madde_header + "\n" + piece).strip() if madde_header else piece
                facets = extract_facets(piece)

                chunk_uid = _stable_uid(source, madde_kind, madde_no, span_idx, chunk_seq, page_start, page_end, piece[:200])

                chunk_obj: Dict[str, Any] = {
                    "source": source,
                    "text": text_out,
                    "embed_text": embed_text,
                    "madde_kind": madde_kind,
                    "madde_no": madde_no,
                    "madde_header": madde_header,
                    "madde": madde_header,
                    "span_index": span_idx,
                    "chunk_seq": chunk_seq,
                    "chunk_uid": chunk_uid,
                    "page_start": page_start,
                    "page_end": page_end,
                    **facets,
                }

                for k in cfg.passthrough_keys:
                    if k in doc:
                        chunk_obj[k] = doc[k]

                all_chunks.append(chunk_obj)

    logger.info("Toplam %d chunk oluşturuldu.", len(all_chunks))
    return all_chunks