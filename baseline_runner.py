from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests


API_URL = "http://127.0.0.1:8000/ask"
OUTPUT_PATH = Path("baseline_results.json")

CASES = [
    {
        "question": "Özel öğrenci kredi toplamı nasıl değerlendirilir?",
        "expected_behavior": "direct",
    },
    {
        "question": "Uzaktan öğretimde toplam AKTS'nin en fazla yüzde kaçı uzaktan olabilir?",
        "expected_behavior": "direct",
    },
    {
        "question": "Ek sınav hangi durumda açılır?",
        "expected_behavior": "direct",
    },
    {
        "question": "Devamsızlık sınırı nedir?",
        "expected_behavior": "clarification",
    },
    {
        "question": "Sınav hakkı kaç tane?",
        "expected_behavior": "clarification",
    },
]


def detect_behavior(data: dict[str, Any]) -> str:
    if data.get("needs_clarification"):
        return "clarification"

    answer = str(data.get("answer") or "").lower()
    sources = data.get("sources") or []

    no_answer_markers = (
        "bilgi bulamadım",
        "net bir hüküm bulunmamaktadır",
        "açık hüküm yok",
    )

    if not sources and any(marker in answer for marker in no_answer_markers):
        return "no_answer"

    return "direct"


def summarize_retrieved(data: dict[str, Any]) -> list[dict[str, Any]]:
    result = []

    for item in (data.get("retrieved") or [])[:3]:
        result.append(
            {
                "source": item.get("source"),
                "madde_no": item.get("madde_no"),
                "score": item.get("score"),
                "dense_score": item.get("dense_score"),
                "lexical_score": item.get("lexical_score"),
            }
        )

    return result


results = []

for index, case in enumerate(CASES, start=1):
    question = case["question"]
    expected = case["expected_behavior"]

    print(f"\n[{index}/{len(CASES)}] Soru gönderiliyor:")
    print(question)

    try:
        response = requests.post(
            API_URL,
            json={"question": question},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        actual = detect_behavior(data)

        result = {
            "question": question,
            "expected_behavior": expected,
            "actual_behavior": actual,
            "passed": actual == expected,
            "http_status": response.status_code,
            "answer": data.get("answer"),
            "needs_clarification": data.get("needs_clarification"),
            "clarification_options": data.get("clarification_options"),
            "sources": data.get("sources"),
            "duration": data.get("duration"),
            "session_id": data.get("session_id"),
            "top_retrieved": summarize_retrieved(data),
            "clarification_followup": None,
        }

        if data.get("needs_clarification"):
            clarification = data.get("clarification_options") or {}
            clarification_type = clarification.get("type")
            options = clarification.get("options") or []

            selected_option = next(
                (
                    option
                    for option in options
                    if option.get("value") == "normal_donem"
                ),
                options[0] if options else None,
            )

            if clarification_type and selected_option:
                followup_payload = {
                    "question": selected_option.get("label"),
                    "session_id": data.get("session_id"),
                    "clarification": {
                        clarification_type: selected_option.get("value")
                    },
                }

                followup_response = requests.post(
                    API_URL,
                    json=followup_payload,
                    timeout=300,
                )
                followup_response.raise_for_status()
                followup_data = followup_response.json()

                result["clarification_followup"] = {
                    "selected_type": clarification_type,
                    "selected_value": selected_option.get("value"),
                    "selected_label": selected_option.get("label"),
                    "answer": followup_data.get("answer"),
                    "needs_clarification": followup_data.get(
                        "needs_clarification"
                    ),
                    "sources": followup_data.get("sources"),
                    "duration": followup_data.get("duration"),
                    "top_retrieved": summarize_retrieved(followup_data),
                }

        results.append(result)

        status = "BAŞARILI" if result["passed"] else "BEKLENTİDEN FARKLI"

        print(
            f"Sonuç: {status} | "
            f"beklenen={expected} | gerçekleşen={actual}"
        )
        print("Cevap:", data.get("answer"))
        print("Kaynaklar:", data.get("sources"))
        print("Süre:", data.get("duration"))

        if data.get("needs_clarification"):
            print(
                "Clarification:",
                data.get("clarification_options"),
            )

            followup = result.get("clarification_followup")
            if followup:
                print(
                    "Clarification sonrası cevap:",
                    followup.get("answer"),
                )

    except Exception as exc:
        results.append(
            {
                "question": question,
                "expected_behavior": expected,
                "actual_behavior": "error",
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

        print("HATA:", type(exc).__name__, exc)


summary = {
    "total": len(results),
    "passed": sum(1 for result in results if result.get("passed")),
    "failed": sum(1 for result in results if not result.get("passed")),
    "results": results,
}

OUTPUT_PATH.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n==============================")
print("BASELINE ÖZETİ")
print("==============================")
print("Toplam:", summary["total"])
print("Beklendiği gibi çalışan:", summary["passed"])
print("Beklentiden farklı çalışan:", summary["failed"])
print("Sonuç dosyası:", OUTPUT_PATH.resolve())

for result in results:
    status = "OK" if result.get("passed") else "SORUN"
    print(
        f"{status} | "
        f"{result.get('expected_behavior')} -> "
        f"{result.get('actual_behavior')} | "
        f"{result.get('question')}"
    )
