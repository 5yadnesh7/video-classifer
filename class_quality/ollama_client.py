"""Ollama Cloud chat client.

Cloud models do not accept video or audio. This client is not used by the
rules-only analyzer. Calls stay disabled unless `permitted=True`, which must
only be set after the organiser allows this service.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Protocol

from pydantic import BaseModel, ValidationError

DEFAULT_MODEL = "gemma4:31b"
CHAT_URL = "https://ollama.com/api/chat"
_MEDIA_SUFFIXES = (".mp4", ".mkv", ".mov", ".webm", ".wav", ".mp3", ".m4a", ".vtt")
_CRITERION_IDS = {
    "opening": "opening",
    "opening and rapport": "opening",
    "inclusion": "inclusion",
    "inclusion and individual engagement": "inclusion",
    "tone": "tone",
    "tone and energy": "tone",
    "closure": "closure",
    "lesson structure and closure": "closure",
    "homework": "homework",
    "homework assigned": "homework",
    "teacher_presence": "teacher_presence",
    "teacher presence": "teacher_presence",
    "teacher presence on camera": "teacher_presence",
    "branding": "branding",
    "professional setup": "branding",
    "technical": "technical",
    "technical quality": "technical",
}


class OllamaConfigError(RuntimeError):
    """The client is not configured to call Ollama Cloud."""


class OllamaResponseError(RuntimeError):
    """The cloud response could not be used after the allowed retry."""


class JudgeCriterion(BaseModel):
    id: str
    rating: int | None
    explanation: str
    timestamps: list[str]
    evidence: str = ""


class JudgeResponse(BaseModel):
    criteria: list[JudgeCriterion]


class Transport(Protocol):
    def post(self, url: str, headers: dict[str, str], payload: dict, timeout: float) -> tuple[int, bytes]:
        """Return an HTTP status and body."""


def urllib_transport(url: str, headers: dict[str, str], payload: dict, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise TimeoutError("Ollama Cloud request timed out.") from error
        raise


class UrllibTransport:
    """The transport the client calls. `urllib_transport` is its request function."""

    def post(self, url: str, headers: dict[str, str], payload: dict, timeout: float) -> tuple[int, bytes]:
        return urllib_transport(url, headers, payload, timeout)


class OllamaCloudClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float = 60,
        transport: Transport = UrllibTransport(),
        permitted: bool = False,
        url: str = CHAT_URL,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.transport = transport
        self.permitted = permitted
        self.url = url
        self._api_key = api_key

    def complete_json(self, prompt: str, images: list[str] | None = None) -> JudgeResponse:
        if not self.permitted:
            raise PermissionError("Ollama Cloud is disabled until the organiser permits it.")
        api_key = self._api_key if self._api_key is not None else os.environ.get("OLLAMA_API_KEY")
        if not api_key:
            raise OllamaConfigError("OLLAMA_API_KEY is not set.")
        self._reject_recording_inputs(images)
        message: dict[str, object] = {"role": "user", "content": prompt}
        if images:
            message["images"] = images
        payload = {"model": self.model, "messages": [message], "stream": False}
        headers = {"Authorization": f"Bearer {api_key}"}
        last_error: OllamaResponseError | None = None
        for _ in range(2):
            try:
                status, body = self.transport.post(self.url, headers, payload, self.timeout)
            except TimeoutError:
                last_error = OllamaResponseError("Ollama Cloud request timed out.")
                continue
            if status in (401, 403):
                raise OllamaResponseError(f"Ollama Cloud request failed with status {status}.")
            if status == 429:
                last_error = OllamaResponseError("Ollama Cloud request failed with status 429.")
                continue
            if status != 200:
                raise OllamaResponseError(f"Ollama Cloud request failed with status {status}.")
            try:
                return self._parse(body)
            except (json.JSONDecodeError, ValidationError, ValueError, KeyError):
                last_error = OllamaResponseError("Ollama Cloud response did not match the expected JSON.")
        assert last_error is not None
        raise last_error

    @staticmethod
    def _reject_recording_inputs(images: list[str] | None) -> None:
        for image in images or []:
            lowered = image.lower()
            looks_like_path = (
                "\\" in image
                or re.match(r"^[A-Za-z]:[\\/]", image) is not None
                or lowered.endswith(_MEDIA_SUFFIXES)
            )
            if looks_like_path:
                raise ValueError("Ollama Cloud accepts text and still images, not recording files.")

    @staticmethod
    def _parse(body: bytes) -> JudgeResponse:
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("response was not a JSON object")
        message = payload["message"]
        if not isinstance(message, dict):
            raise ValueError("response did not contain a message")
        content = _message_text(message.get("content"))
        return JudgeResponse.model_validate(_normalize_judge_payload(_json_object(content)))


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content") or ""
                parts.append(text if isinstance(text, str) else "")
        return "\n".join(parts)
    raise ValueError("response content was not text")


def _criterion_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return _CRITERION_IDS.get(re.sub(r"\s+", " ", value.strip().lower()))


def _rating(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int) and value in (0, 1, 2):
        return value
    if isinstance(value, float) and value in (0.0, 1.0, 2.0):
        return int(value)
    if isinstance(value, str):
        match = re.fullmatch(r"([012])(?:\s*/\s*2)?", value.strip())
        if match:
            return int(match.group(1))
    return None


def _timestamps(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item).strip()]
    return []


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _criteria_items(payload: object) -> list[dict] | None:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return None
    criteria = payload.get("criteria")
    if isinstance(criteria, list):
        return [item for item in criteria if isinstance(item, dict)]
    if isinstance(criteria, dict):
        items = []
        for key, value in criteria.items():
            if isinstance(value, dict):
                items.append({"id": key, **value})
            else:
                items.append({"id": key, "rating": value})
        return items
    for key in ("ai_result", "result", "assessment", "judgement", "judgment"):
        nested = payload.get(key)
        if nested is not None and nested is not payload:
            found = _criteria_items(nested)
            if found:
                return found
    return None


def _normalize_judge_payload(payload: object) -> object:
    """Accept the JSON shapes Gemma returns after it has been shown the rules."""
    items = _criteria_items(payload)
    if not items:
        return payload
    criteria = []
    for item in items:
        criterion_id = _criterion_id(item.get("id") or item.get("criterion") or item.get("name"))
        if criterion_id is None:
            continue
        criteria.append(
            {
                "id": criterion_id,
                "rating": _rating(item.get("rating")),
                "explanation": _text(item.get("explanation")),
                "timestamps": _timestamps(item.get("timestamps")),
                "evidence": _text(item.get("evidence")),
            }
        )
    if not criteria:
        return payload
    return {"criteria": criteria}


def _json_object(content: str) -> object:
    text = content.strip()
    if text.startswith("```"):
        parts = text.split("\n", 1)
        text = parts[1].rsplit("```", 1)[0] if len(parts) == 2 else text
    text = re.sub(r",\s*([}\]])", r"\1", text)
    decoder = json.JSONDecoder()
    found: list[object] = []
    index = 0
    while True:
        start = text.find("{", index)
        array_start = text.find("[", index)
        if array_start >= 0 and (start < 0 or array_start < start):
            start = array_start
        if start < 0:
            break
        try:
            value, offset = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        found.append(value)
        index = start + offset
    containers = [value for value in found if _criteria_items(value)]
    if containers:
        return max(containers, key=lambda value: len(_criteria_items(value) or []))
    items = [
        value
        for value in found
        if isinstance(value, dict) and _criterion_id(value.get("id") or value.get("criterion") or value.get("name"))
    ]
    if items:
        return items
    if found:
        return found[0]
    raise ValueError("response did not contain a JSON object")
