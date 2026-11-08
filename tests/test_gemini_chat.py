import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.server import app


def test_live_chat_forwards_turns_including_questions_and_repeated_prompts(gemini_transport):
    client = TestClient(app)
    endpoint = "/api/hypothesis/interpret"
    history = []
    first = "Do Coke and Pepsi move together?"
    for text in [first, "Change the holding period to 7 trading days", first]:
        response = client.post(endpoint, json={"hypothesis_text": text, "chat_history": history})
        assert response.status_code == 200, response.text
        result = response.json()
        spec = result["proposed_spec"]
        assert spec["template_id"] == "pairs_cointegration_reversion"
        assert (spec["asset_a"], spec["asset_b"]) == ("KO", "PEP")
        assert spec["human_confirmed"] is False
        if "holding" in text:
            assert spec["parameters"]["holding_period"] == 7
        assert result["interpretation_mode"] == "live_gemini"
        assert result["model_used"] == "gemini-test-version"
        inputs = json.loads(gemini_transport[-1]["contents"][0]["parts"][0]["text"])
        assert inputs == [
            {"input": t["content"], "basket": t["basket"]} for t in history if t["role"] == "user"
        ] + [{"input": text}]
        history = result["chat_history"]
        assert history[-2]["basket"] == spec["template_id"]
        assert history[-1]["content"] == result["gemini_reply"]
    assert len(gemini_transport) == 3


def test_unconfigured_chat_fails_explicitly(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    response = TestClient(app).post("/api/hypothesis/interpret", json={"hypothesis_text": "hello"})
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


@pytest.mark.parametrize(("status", "payload", "expected"), [
    (429, {"error": {"code": 429, "message": "quota"}}, 429),
    (200, {"candidates": [{"finishReason": "STOP", "content": {"parts": []}}]}, 502),
    (200, {"promptFeedback": {"blockReason": "SAFETY"}}, 502),
])
def test_no_synthetic_reply_on_provider_failure(monkeypatch, status, payload, expected):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, req, **kw: httpx.Response(
        status, request=req, json=payload))
    response = TestClient(app).post("/api/hypothesis/interpret", json={"hypothesis_text": "Do KO and PEP move together?"})
    assert response.status_code == expected
    assert "detail" in response.json()
    assert "gemini_reply" not in response.json()


@pytest.mark.parametrize("gemini_basket", [
    "return_threshold", "consecutive_streak", "mean_reversion_oscillator", "trend_breakout",
    "pairs_cointegration_reversion", "no_match", "needs_detail", "invented_model",
], indirect=True)
def test_only_catalogued_basket_can_reach_configuration(gemini_basket, gemini_transport):
    response = TestClient(app).post("/api/hypothesis/interpret", json={"hypothesis_text": "Test AAPL price movements"})
    if gemini_basket == "invented_model":
        assert response.status_code == 502
        return
    assert response.status_code == 200
    result = response.json()
    if gemini_basket in ("needs_detail", "no_match"):
        assert result["proposed_spec"] is None
        assert result["steered_cards"] == []
    else:
        assert result["proposed_spec"]["template_id"] == gemini_basket
        assert result["proposed_spec"]["human_confirmed"] is False


@pytest.mark.parametrize("gemini_basket", ["consecutive_streak"], indirect=True)
@pytest.mark.parametrize(("initial", "clarification", "asset"), [
    ("I want to test Apple.", "After three consecutive down days.", "AAPL"),
    ("After three consecutive down days.", "Apple", "AAPL"),
    ("After three consecutive down days.", "KO", "KO"),
])
def test_clarified_asset_survives_later_refinements(gemini_basket, gemini_transport, initial, clarification, asset):
    client = TestClient(app)
    history = [{"role": "user", "content": initial, "basket": "needs_detail"}]
    for text in [clarification, "Hold for 7 trading days"]:
        resp = client.post("/api/hypothesis/interpret", json={"hypothesis_text": text, "chat_history": history})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        spec = data["proposed_spec"]
        assert spec["asset_a"] == asset
        assert spec["template_id"] == "consecutive_streak"
        assert spec["parameters"]["streak_length"] == 3
        history = data["chat_history"]
        assert history[-2]["content"] == text
    assert spec["parameters"]["holding_period"] == 7
