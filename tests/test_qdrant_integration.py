from __future__ import annotations

import pytest

from qdrant_service import (
    COLLECTION_NAME,
    QDRANT_URL,
    get_client,
)


pytestmark = pytest.mark.integration

EXPECTED_POINT_COUNT = 748
EXPECTED_VECTOR_SIZE = 1024

REQUIRED_PAYLOAD_KEYS = {
    "source",
    "source_url",
    "text",
    "chunk_uid",
    "embedding_model",
    "ext",
    "path",
    "madde_kind",
    "topic",
    "topic_domain",
}


def _value_as_lower_text(value: object) -> str:
    enum_value = getattr(value, "value", value)
    return str(enum_value).lower()


def test_qdrant_collection_matches_expected_configuration() -> None:
    # Arrange
    client = get_client()

    # Act
    collection_info = client.get_collection(
        collection_name=COLLECTION_NAME,
    )

    vectors = collection_info.config.params.vectors

    # Assert
    assert QDRANT_URL
    assert COLLECTION_NAME == "mevzuat_rag"

    assert (
        _value_as_lower_text(collection_info.status)
        == "green"
    )

    assert collection_info.points_count == EXPECTED_POINT_COUNT

    assert vectors.size == EXPECTED_VECTOR_SIZE

    assert (
        _value_as_lower_text(vectors.distance)
        == "cosine"
    )


def test_qdrant_points_contain_required_payload_fields() -> None:
    # Arrange
    client = get_client()

    # Act
    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=5,
        with_payload=True,
        with_vectors=False,
    )

    # Assert
    assert records

    for record in records:
        payload = record.payload or {}
        if "madde_no" in payload:
            assert isinstance(payload["madde_no"], int)
            assert payload["madde_no"] > 0
        

        assert REQUIRED_PAYLOAD_KEYS.issubset(
            payload.keys()
        )

        assert isinstance(payload["source"], str)
        assert payload["source"].strip()

        assert isinstance(payload["text"], str)
        assert payload["text"].strip()

        assert isinstance(payload["chunk_uid"], str)
        assert payload["chunk_uid"].strip()

        assert isinstance(payload["embedding_model"], str)
        assert payload["embedding_model"].strip()