from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from chunker import ChunkConfig, chunk_mevzuat
from file_reader import ReadConfig, read_all_documents

logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "mevzuat_rag")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "ytu-ce-cosmos/turkish-e5-large")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024"))
UUID_NS = uuid.NAMESPACE_URL

KEYWORD_INDEX = [
    "source",
    "term_scope",
    "program_level",
    "student_status",
    "teaching_mode",
    "topic",
    "topic_domain",
    "madde_kind",
    "ext",
]

INT_INDEX = ["madde_no", "page_start", "page_end", "mtime", "size_bytes", "pdf_pages"]
BOOL_INDEX = ["contains_akts", "has_tezli", "has_tezsiz", "pdf_low_text"]


def uuid5_from_chunk_uid(chunk_uid: str) -> str:
    return str(uuid.uuid5(UUID_NS, f"{COLLECTION_NAME}:{chunk_uid}"))


def clean_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in payload.items() if v is not None}


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    if recreate:
        logger.warning("Collection yeniden oluşturuluyor: %s", COLLECTION_NAME)
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
        return

    try:
        client.get_collection(COLLECTION_NAME)
        logger.info("Collection zaten mevcut: %s", COLLECTION_NAME)
    except Exception:
        logger.info("Collection oluşturuluyor: %s", COLLECTION_NAME)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )


def create_payload_indexes(client: QdrantClient) -> None:
    for field_name in KEYWORD_INDEX:
        try:
            client.create_payload_index(
                COLLECTION_NAME,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception:
            logger.debug("Keyword index atlandı: %s", field_name)

    for field_name in INT_INDEX:
        try:
            client.create_payload_index(
                COLLECTION_NAME,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.INTEGER,
                wait=True,
            )
        except Exception:
            logger.debug("Integer index atlandı: %s", field_name)

    for field_name in BOOL_INDEX:
        try:
            client.create_payload_index(
                COLLECTION_NAME,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.BOOL,
                wait=True,
            )
        except Exception:
            logger.debug("Bool index atlandı: %s", field_name)


def build_chunks() -> List[Dict[str, Any]]:
    files_dir = os.getenv("FILES_DIR", "files")
    docs = read_all_documents(files_dir=files_dir, cfg=ReadConfig())
    return chunk_mevzuat(docs, cfg=ChunkConfig())


def upsert_all(
    client: QdrantClient,
    chunks: List[Dict[str, Any]],
    embed_batch: int = 64,
    upsert_batch: int = 256,
) -> None:
    model = SentenceTransformer(EMBED_MODEL)
    buffer: List[models.PointStruct] = []

    def flush() -> None:
        nonlocal buffer
        if buffer:
            client.upsert(collection_name=COLLECTION_NAME, points=buffer, wait=True)
            logger.info("%d kayıt upsert edildi.", len(buffer))
            buffer = []

    for i in range(0, len(chunks), embed_batch):
        batch = chunks[i : i + embed_batch]
        texts = [f"passage: {(c.get('embed_text') or c.get('text', '')).strip()}" for c in batch]
        vectors = model.encode(
            texts,
            batch_size=min(embed_batch, len(texts)),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        for chunk, vector in zip(batch, vectors):
            point_id = uuid5_from_chunk_uid(chunk["chunk_uid"])
            payload = clean_payload(
                {
                    "text": chunk.get("text"),
                    "source": chunk.get("source"),
                    "source_url": chunk.get("source_url"),
                    "chunk_uid": chunk.get("chunk_uid"),
                    "madde_kind": chunk.get("madde_kind"),
                    "madde_no": chunk.get("madde_no"),
                    "madde_header": chunk.get("madde_header"),
                    "madde": chunk.get("madde"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "term_scope": chunk.get("term_scope"),
                    "program_level": chunk.get("program_level"),
                    "student_status": chunk.get("student_status"),
                    "teaching_mode": chunk.get("teaching_mode"),
                    "topic_domain": chunk.get("topic_domain"),
                    "topic": chunk.get("topic"),
                    "topic_candidates": chunk.get("topic_candidates"),
                    "akts_values": chunk.get("akts_values"),
                    "contains_akts": chunk.get("contains_akts"),
                    "has_tezli": chunk.get("has_tezli"),
                    "has_tezsiz": chunk.get("has_tezsiz"),
                    "ext": chunk.get("ext"),
                    "mtime": chunk.get("mtime"),
                    "size_bytes": chunk.get("size_bytes"),
                    "pdf_pages": chunk.get("pdf_pages"),
                    "pdf_low_text": chunk.get("pdf_low_text"),
                    "sha256": chunk.get("sha256"),
                    "path": chunk.get("path"),
                    "embedding_model": EMBED_MODEL,
                }
            )

            buffer.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

            if len(buffer) >= upsert_batch:
                flush()

    flush()


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    client = get_client()

    chunks = build_chunks()
    logger.info("Toplam chunk: %d", len(chunks))

    ensure_collection(
        client,
        recreate=os.getenv("RECREATE_COLLECTION", "false").lower() == "true",
    )
    create_payload_indexes(client)
    upsert_all(
        client,
        chunks,
        embed_batch=int(os.getenv("EMBED_BATCH", "64")),
        upsert_batch=int(os.getenv("UPSERT_BATCH", "256")),
    )

    logger.info("İndeksleme tamamlandı.")


if __name__ == "__main__":
    main()