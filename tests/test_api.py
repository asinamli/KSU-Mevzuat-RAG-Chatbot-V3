from fastapi.testclient import TestClient
import pytest

import api


client = TestClient(api.app)


@pytest.fixture(autouse=True)
def clear_sessions():
    api.sessions.clear()
    yield
    api.sessions.clear()


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    body = response.json()
    assert body["message"] == "KSÜ Mevzuat RAG API Çalışıyor!"
    assert body["version"] == "2.2.0"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] in {"healthy", "degraded"}
    assert isinstance(body["rag_ready"], bool)
    assert body["model"] == api.rag_llm.MODEL_NAME
    assert body["collection"] == api.rag_llm.COLLECTION_NAME


def test_ask_endpoint_returns_rag_answer(monkeypatch):
    def fake_ask(**kwargs):
        return (
            "Test cevabı",
            [],
            0.12,
            [],
        )

    monkeypatch.setattr(api.rag_llm, "ask", fake_ask)

    response = client.post(
        "/ask",
        json={"question": "Test sorusu"},
    )

    assert response.status_code == 200

    body = response.json()
    assert body["answer"] == "Test cevabı"
    assert body["needs_clarification"] is False
    assert body["session_id"]


def test_ask_rejects_empty_question():
    response = client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_clarify_requires_clarification_parameter():
    response = client.post(
        "/ask/clarify",
        json={"question": "Test sorusu"},
    )

    assert response.status_code == 400


def test_session_history_and_delete(monkeypatch):
    def fake_ask(**kwargs):
        return (
            "Test cevabı",
            [],
            0.10,
            [],
        )

    monkeypatch.setattr(api.rag_llm, "ask", fake_ask)

    ask_response = client.post(
        "/ask",
        json={
            "question": "Test sorusu",
            "session_id": "test-session",
        },
    )

    assert ask_response.status_code == 200

    history_response = client.get(
        "/session/test-session/history"
    )

    assert history_response.status_code == 200
    assert history_response.json()["history"] == [
        {"role": "user", "content": "Test sorusu"},
        {"role": "assistant", "content": "Test cevabı"},
    ]

    delete_response = client.delete(
        "/session/test-session"
    )

    assert delete_response.status_code == 200

    history_after_delete = client.get(
        "/session/test-session/history"
    )

    assert history_after_delete.json()["history"] == []