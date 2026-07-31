from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from PyPDF2 import PdfReader
from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

try:
    from source_url_map import SOURCE_URL_MAP
except Exception:
    SOURCE_URL_MAP = {}

logger = logging.getLogger(__name__)

FILES_DIR = Path("files")

_SOFT_HYPHEN = "\u00ad"
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)", flags=re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class ReadConfig:
    recursive: bool = True
    include_pdf_page_markers: bool = True
    include_docx_table_markers: bool = True
    max_pdf_pages: Optional[int] = None
    min_pdf_chars_warn: int = 80
    compute_sha256: bool = False


def _clean_extracted_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(_SOFT_HYPHEN, "")
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pdf(path: Union[str, Path], cfg: ReadConfig = ReadConfig()) -> Dict[str, object]:
    path = Path(path)
    parts: List[str] = []
    pages_read = 0

    try:
        reader = PdfReader(str(path), strict=False)

        for i, page in enumerate(reader.pages):
            if cfg.max_pdf_pages is not None and i >= cfg.max_pdf_pages:
                break

            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            page_text = _clean_extracted_text(page_text)
            pages_read += 1

            if cfg.include_pdf_page_markers:
                parts.append(f"\n\n[PDF_PAGE {i + 1}]\n{page_text}")
            else:
                parts.append(page_text)

    except Exception as e:
        return {
            "text": "",
            "pdf_pages": 0,
            "pdf_low_text": True,
            "pdf_error": f"{type(e).__name__}: {e}",
        }

    full_text = "\n".join([p for p in parts if p is not None]).strip()
    return {
        "text": full_text,
        "pdf_pages": pages_read,
        "pdf_low_text": len(full_text) < cfg.min_pdf_chars_warn,
    }


def _iter_block_items(parent: Union[DocxDocument, _Cell]):
    if isinstance(parent, DocxDocument):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise TypeError(f"Unsupported parent type: {type(parent)}")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def _extract_cell_text(cell: _Cell, cfg: ReadConfig) -> str:
    parts: List[str] = []

    if hasattr(cell, "iter_inner_content"):
        for item in cell.iter_inner_content():
            if isinstance(item, Paragraph):
                txt = _clean_extracted_text(item.text)
                if txt:
                    parts.append(txt)
            elif isinstance(item, Table):
                ttxt = _extract_table_text(item, cfg)
                if ttxt:
                    parts.append(ttxt)
        return "\n".join(parts).strip()

    for p in cell.paragraphs:
        txt = _clean_extracted_text(p.text)
        if txt:
            parts.append(txt)

    for t in getattr(cell, "tables", []):
        ttxt = _extract_table_text(t, cfg)
        if ttxt:
            parts.append(ttxt)

    return "\n".join(parts).strip()


def _extract_table_text(table: Table, cfg: ReadConfig) -> str:
    rows_out: List[str] = []

    for row in table.rows:
        cell_texts: List[str] = []
        for cell in row.cells:
            t = _extract_cell_text(cell, cfg)
            t = _clean_extracted_text(t).replace("\n", " ").strip()
            if t:
                cell_texts.append(t)

        if cell_texts:
            rows_out.append(" | ".join(cell_texts))

    table_text = "\n".join(rows_out).strip()
    if not table_text:
        return ""

    if cfg.include_docx_table_markers:
        return f"\n[DOCX_TABLE]\n{table_text}\n[/DOCX_TABLE]\n"
    return table_text


def read_docx(path: Union[str, Path], cfg: ReadConfig = ReadConfig()) -> str:
    path = Path(path)
    doc = Document(str(path))

    out: List[str] = []
    for block in _iter_block_items(doc):
        if isinstance(block, Paragraph):
            txt = _clean_extracted_text(block.text)
            if txt:
                out.append(txt)
        elif isinstance(block, Table):
            ttxt = _extract_table_text(block, cfg)
            if ttxt:
                out.append(ttxt)

    return "\n".join(out).strip()


def _list_input_files(files_dir: Path, cfg: ReadConfig) -> List[Path]:
    exts = {".pdf", ".docx"}

    if cfg.recursive:
        paths = [
            p
            for p in files_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in exts
            and not p.name.startswith("~$")
        ]
    else:
        paths = [
            p
            for p in files_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in exts
            and not p.name.startswith("~$")
        ]

    return sorted(paths, key=lambda p: str(p).lower())


def read_all_documents(
    files_dir: Union[str, Path] = FILES_DIR,
    cfg: ReadConfig = ReadConfig(),
) -> List[Dict[str, object]]:
    files_dir = Path(files_dir)
    documents: List[Dict[str, object]] = []

    if not files_dir.exists():
        logger.warning("'%s' klasörü bulunamadı.", files_dir)
        return []

    for path in _list_input_files(files_dir, cfg):
        rel = path.relative_to(files_dir)
        source = str(rel).replace("\\", "/")

        meta: Dict[str, object] = {
            "source": source,
            "source_url": SOURCE_URL_MAP.get(source),
            "path": str(path),
            "ext": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "mtime": int(path.stat().st_mtime),
        }

        text = ""
        if path.suffix.lower() == ".pdf":
            res = read_pdf(path, cfg)
            text = str(res.get("text", "") or "")
            meta["pdf_pages"] = res.get("pdf_pages")
            meta["pdf_low_text"] = res.get("pdf_low_text")
            if res.get("pdf_error"):
                meta["pdf_error"] = res.get("pdf_error")

        elif path.suffix.lower() == ".docx":
            text = read_docx(path, cfg)

        text = text.strip()
        if not text:
            logger.warning("Boş içerik nedeniyle dosya atlandı: %s", source)
            continue

        doc_obj: Dict[str, object] = {"source": meta["source"], "text": text, **meta}
        if cfg.compute_sha256:
            doc_obj["sha256"] = _sha256_file(path)

        documents.append(doc_obj)

    logger.info("Toplam %d belge okundu.", len(documents))
    return documents