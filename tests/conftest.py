from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
REAL_DOCUMENTS_DIR = PROJECT_ROOT / "backend" / "files"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture(scope="session")
def real_documents_dir() -> Path:
    if not REAL_DOCUMENTS_DIR.is_dir():
        pytest.fail(
            f"Gerçek belge klasörü bulunamadı: {REAL_DOCUMENTS_DIR}",
            pytrace=False,
        )

    return REAL_DOCUMENTS_DIR