from config import Settings


def test_settings_load_default_values(monkeypatch):
    env_names = (
        "LLM_MODEL",
        "QDRANT_URL",
        "QDRANT_COLLECTION",
        "EMBED_MODEL",
    )

    for name in env_names:
        monkeypatch.delenv(name, raising=False)

    settings = Settings()

    assert settings.LLM_MODEL == "gemma2:latest"
    assert settings.QDRANT_URL == "http://localhost:6333"
    assert settings.QDRANT_COLLECTION == "mevzuat_rag"
    assert settings.EMBED_MODEL == "ytu-ce-cosmos/turkish-e5-large"


def test_settings_load_environment_overrides(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6334")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_collection")
    monkeypatch.setenv("EMBED_MODEL", "test-embedding-model")

    settings = Settings()

    assert settings.LLM_MODEL == "test-model"
    assert settings.QDRANT_URL == "http://localhost:6334"
    assert settings.QDRANT_COLLECTION == "test_collection"
    assert settings.EMBED_MODEL == "test-embedding-model"