"""Minimal Ollama client. stdlib only.

Reliability posture (see DESIGN.md): retry with backoff, salvage malformed
JSON once, and never let a single bad call kill an overnight run — callers
get an OllamaError only after all recovery has failed, and the loop layer
converts that into a flag, not a crash.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request


class OllamaError(Exception):
    pass


def extract_json(text: str):
    """Salvage a JSON object from model text (bare, fenced, or embedded)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    brace = text.find("{")
    if brace != -1:
        depth = 0
        for i in range(brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


class Ollama:
    def __init__(self, host: str | None = None, timeout: int = 900):
        base = host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        if not base.startswith("http"):
            base = "http://" + base
        self.base = base.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def chat(
        self,
        model: str,
        messages: list[dict],
        schema: dict | None = None,
        tools: list[dict] | None = None,
        temperature: float = 0.2,
        num_ctx: int = 8192,
        retries: int = 3,
        think: bool | None = None,
    ) -> dict:
        """One chat call. Returns the response `message` dict
        ({role, content, [tool_calls]}). Retries transport errors with backoff.

        think=False disables hidden reasoning on thinking-capable models — without
        it, qwen3.5:4b spent 2m16s of chain-of-thought on a one-word answer
        (measured 2026-07-12; the June image-renamer trap, same shape). Only send
        the key for models that support thinking: Ollama 400s otherwise.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        if think is not None:
            payload["think"] = think
        if schema is not None:
            payload["format"] = schema
        if tools is not None:
            payload["tools"] = tools

        last_err = None
        for attempt in range(retries):
            try:
                data = self._post("/api/chat", payload)
                return data["message"]
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError) as e:
                last_err = e
                time.sleep(min(2**attempt * 5, 60))
        raise OllamaError(f"chat failed after {retries} attempts: {last_err}")

    def chat_json(self, model: str, messages: list[dict], schema: dict, **kw) -> dict:
        """Chat expecting a schema-constrained JSON object; salvages, then
        re-asks once with an explicit instruction before giving up."""
        msg = self.chat(model, messages, schema=schema, **kw)
        parsed = extract_json(msg.get("content") or "")
        if parsed is not None:
            return parsed
        retry_messages = messages + [
            {"role": "assistant", "content": msg.get("content") or ""},
            {"role": "user", "content": "That was not valid JSON. Respond with ONLY the JSON object."},
        ]
        msg = self.chat(model, retry_messages, schema=schema, **kw)
        parsed = extract_json(msg.get("content") or "")
        if parsed is None:
            raise OllamaError(f"unparseable JSON from {model}")
        return parsed
