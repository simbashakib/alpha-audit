"""Live Gemini conversation for the web workbench."""

import json
import os
from urllib.parse import quote

import httpx

from engine.hypothesis_parser import interpret_hypothesis
from engine.schemas import ChatTurn, InterpretationResponse


class GeminiChatError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def chat_about_hypothesis(
    text: str, history: list[ChatTurn], baskets: dict[str, str],
) -> InterpretationResponse:
    instructions = (
        "Classify the latest user input into one of Alpha Auditor's predefined strategy baskets. "
        "Use earlier user inputs to resolve follow-up changes. Choose needs_detail when the trigger "
        "is ambiguous or the input is only a greeting or question about the interface. Choose no_match "
        "when the requested signal needs unsupported data, a custom strategy, or multiple incompatible "
        "baskets. Do not force a match. Comparing two stocks for co-movement belongs in the pairs basket. "
        "Return only the classification. Available baskets: " + json.dumps(baskets)
    )
    inputs = [
        {"input": turn.content, "basket": turn.basket}
        for turn in history if turn.role == "user"
    ]
    inputs.append({"input": text})
    schema = {
        "type": "OBJECT", "properties": {
            "basket": {"type": "STRING", "enum": [*baskets, "needs_detail", "no_match"]},
        }, "required": ["basket"],
    }
    contents = [{"role": "user", "parts": [{"text": json.dumps(inputs)}]}]
    raw, model = generate_gemini_content(instructions, contents, schema)
    try:
        decision = json.loads(raw)
        basket = decision["basket"]
        if set(decision) != {"basket"} or basket not in [*baskets, "needs_detail", "no_match"]:
            raise ValueError("Unknown classification")
    except (ValueError, TypeError, KeyError) as exc:
        raise GeminiChatError(502, "Gemini returned an invalid basket. Try submitting the hypothesis again.") from exc

    if basket == "no_match":
        reason = "This hypothesis falls outside the five supported baskets. Try a rule based on daily prices or on the relationship between two assets."
        result = InterpretationResponse(status="unsupported", gemini_reply=reason, unsupported_reason=reason)
    elif basket == "needs_detail":
        basket = "needs_detail"
        result = InterpretationResponse(status="needs_clarification", gemini_reply=(
            "Which asset or pair do you want to test, and what price movement would trigger the test?"
        ))
    else:
        result = interpret_hypothesis(text, history, template=basket)
        if result.proposed_spec is None:
            basket = "needs_detail" if result.status == "needs_clarification" else "no_match"
        elif result.proposed_spec.template_id != basket:
            raise GeminiChatError(502, "The selected basket could not be configured. Try rephrasing your hypothesis.")

    result.model_used = model
    result.interpretation_mode = "live_gemini"
    result.chat_history = [*history, ChatTurn(role="user", content=text, basket=basket),
                           ChatTurn(role="assistant", content=result.gemini_reply)]
    return result


def generate_gemini_content(
    instructions: str, contents: list[dict], response_schema: dict,
) -> tuple[str, str]:
    key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        raise GeminiChatError(503, "Gemini chat needs a server API key. Set GEMINI_API_KEY and restart the server.")

    model = os.getenv("ALPHA_AUDITOR_GEMINI_MODEL", "").strip() or "gemini-3.5-flash"
    payload = {"systemInstruction": {"parts": [{"text": instructions}]}, "contents": contents}
    payload["generationConfig"] = {"responseMimeType": "application/json", "responseSchema": response_schema}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe='')}:generateContent"
    try:
        response = httpx.post(
            url, headers={"x-goog-api-key": key}, timeout=30.0,
            json=payload,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            raise GeminiChatError(429, "Gemini’s request limit was reached. Please try again later.") from exc
        if exc.response.status_code in (400, 401, 403, 404):
            raise GeminiChatError(502, "Gemini rejected the request. Check the server API key and model access.") from exc
        raise GeminiChatError(503, "Gemini is unavailable. Please try sending your message again.") from exc
    except httpx.RequestError as exc:
        raise GeminiChatError(503, "The server could not reach Gemini. Please try again.") from exc

    try:
        data = response.json()
        candidate = data["candidates"][0]
        if candidate.get("finishReason") != "STOP":
            raise GeminiChatError(502, "Gemini did not return a complete reply. Try rephrasing your message.")
        reply = "".join(
            part.get("text", "") for part in candidate["content"]["parts"] if not part.get("thought")
        ).strip()
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise GeminiChatError(502, "Gemini did not return a usable reply. Try rephrasing your message.") from exc
    if not reply:
        raise GeminiChatError(502, "Gemini returned an empty reply. Try sending your message again.")

    return reply, data.get("modelVersion") or model
