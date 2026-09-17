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
    retrieved = [
        {
            "source": "test-belge.pdf",
            "source_url": "https://example.com/test-belge.pdf",
            "text": "Test içerik",
        }
    ]

    def fake_ask(**kwargs):
        return (
            "Test cevabı",
            ["test-belge.pdf"],
            0.126,
            retrieved,
        )

    monkeypatch.setattr(api.rag_llm, "ask", fake_ask)

    response = client.post(
        "/ask",
        json={"question": "Test sorusu"},
    )

    assert response.status_code == 200

    body = response.json()

    assert set(body) == {
        "answer",
        "sources",
        "source_links",
        "duration",
        "retrieved",
        "session_id",
        "needs_clarification",
        "clarification_options",
    }
    assert body["answer"] == "Test cevabı"
    assert body["sources"] == ["test-belge.pdf"]
    assert body["source_links"] == [
        {
            "source": "test-belge.pdf",
            "url": "https://example.com/test-belge.pdf",
        }
    ]
    assert body["duration"] == 0.13
    assert body["retrieved"] == retrieved
    assert body["session_id"]
    assert body["needs_clarification"] is False
    assert body["clarification_options"] is None


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


def test_ask_endpoint_returns_clarification(monkeypatch):
    def fake_ask(**kwargs):
        return (
            f"{api.rag_llm.CLARIFY_PREFIX}|term_scope|normal_donem,yaz_okulu|Hangi dönem için soruyorsunuz?",
            [],
            0.10,
            [],
        )

    monkeypatch.setattr(api.rag_llm, "ask", fake_ask)

    response = client.post(
        "/ask",
        json={
            "question": "Kaç AKTS alabilirim?",
            "session_id": "clarification-session",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] == "Hangi dönem için soruyorsunuz?"
    assert body["sources"] == []
    assert body["source_links"] == []
    assert body["duration"] == 0.10
    assert body["retrieved"] == []
    assert body["session_id"] == "clarification-session"
    assert body["needs_clarification"] is True
    assert body["clarification_options"] == {
        "type": "term_scope",
        "message": "Hangi dönem için soruyorsunuz?",
        "options": [
            {
                "value": "normal_donem",
                "label": "Normal dönem",
            },
            {
                "value": "yaz_okulu",
                "label": "Yaz öğretimi",
            },
        ],
    }


def test_ask_applies_clarification_to_original_question(monkeypatch):
    calls = []

    def fake_ask(**kwargs):
        calls.append(kwargs)

        if len(calls) == 1:
            return (
                f"{api.rag_llm.CLARIFY_PREFIX}|term_scope|normal_donem,yaz_okulu|Hangi dönem için soruyorsunuz?",
                [],
                0.10,
                [],
            )

        return (
            "Yaz öğretimi için test cevabı",
            [],
            0.20,
            [],
        )

    monkeypatch.setattr(api.rag_llm, "ask", fake_ask)

    first_response = client.post(
        "/ask",
        json={
            "question": "Kaç AKTS alabilirim?",
            "session_id": "clarification-session",
        },
    )

    assert first_response.status_code == 200
    assert first_response.json()["needs_clarification"] is True

    second_response = client.post(
        "/ask",
        json={
            "question": "yaz_okulu",
            "session_id": "clarification-session",
            "clarification": {
                "term_scope": "yaz_okulu",
            },
        },
    )

    assert second_response.status_code == 200

    body = second_response.json()

    assert body["answer"] == "Yaz öğretimi için test cevabı"
    assert body["session_id"] == "clarification-session"
    assert body["needs_clarification"] is False
    assert body["clarification_options"] is None

    assert calls[1]["question"] == "Kaç AKTS alabilirim?"
    assert calls[1]["filter_params"] == {
        "term_scope": "yaz_okulu",
    }

