from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"


def load_env(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_deepseek_api_key() -> str:
    load_env()
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY".lower())
    if not key or key.strip() in {"your_key_here", "YOUR_KEY_HERE", "replace_me"}:
        raise RuntimeError("Missing DeepSeek API key. Add DEEPSEEK_API_KEY=... to .env or the environment.")
    return key


def prompt_hash(prompt: str, *, model: str) -> str:
    return hashlib.sha256(f"{model}\n{prompt}".encode("utf-8")).hexdigest()


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object from LLM response.")
    return obj


class DeepSeekClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        api_url: str = DEEPSEEK_API_URL,
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or get_deepseek_api_key()
        self.model = model
        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max_retries

    def chat_json(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 1600) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a careful Lean theorem proving retrieval assistant. Return strict JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.api_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                response_obj = json.loads(raw)
                content = response_obj["choices"][0]["message"]["content"]
                return extract_json_object(content), response_obj
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"DeepSeek request failed after retries: {last_error}") from last_error
