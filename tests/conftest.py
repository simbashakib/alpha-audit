import json

import httpx
import pytest


@pytest.fixture
def gemini_basket(request):
    return getattr(request, "param", "pairs_cointegration_reversion")


@pytest.fixture
def gemini_transport(monkeypatch, gemini_basket):
    """Substitute only the outbound Gemini HTTP transport."""
    calls = []
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    monkeypatch.setenv("ALPHA_AUDITOR_GEMINI_MODEL", "gemini-test")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    def send(client, request, **kwargs):
        assert request.url.host == "generativelanguage.googleapis.com"
        assert request.url.path.endswith("/models/gemini-test:generateContent")
        assert request.headers["x-goog-api-key"] == "test-only"
        body = json.loads(request.content)
        calls.append(body)
        reply = json.dumps({"basket": gemini_basket})
        return httpx.Response(200, request=request, json={
            "candidates": [{"content": {"role": "model", "parts": [{"text": reply}]},
                            "finishReason": "STOP"}],
            "modelVersion": "gemini-test-version",
        })

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", send)
    return calls
