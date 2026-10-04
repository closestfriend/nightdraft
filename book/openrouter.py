"""OpenRouter client: same duck-typed interface as Ollama (chat / chat_json), so
run_loop and plan need no changes. Inherits chat_json (schema salvage + one
re-ask) and overrides only the transport.

Deliberate deviation from DESIGN.md's local-first stance (cloud prose was out of
scope for v1): this is a proxy for the idle-hardware goal. It exercises the
pipeline logic with a stronger model; it says nothing about CPU throughput on
`wer` or about the brie7b voice.
"""

import http.client
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from .ollama import Ollama, OllamaError

DEFAULT_BASE = "https://openrouter.ai/api/v1"


class BudgetExceeded(OllamaError):
    """Spend cap reached. Subclasses OllamaError so run_loop halts the run
    (scene stays pending) instead of writing placeholders."""


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """The harness's tool loop sends bare {"role": "tool", "content": ...}
    results; OpenAI-style APIs reject a tool message without tool_call_id.
    Pair each result with the preceding assistant message's call ids, in order."""
    out, pending_ids = [], []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            pending_ids = [c.get("id") for c in m["tool_calls"]]
            out.append(m)
        elif m.get("role") == "tool" and "tool_call_id" not in m and pending_ids:
            out.append({**m, "tool_call_id": pending_ids.pop(0)})
        else:
            out.append(m)
    return out


def _permanent(code) -> bool:
    """4xx (bad key, no credits, bad model...) will not fix itself; everything
    else — 408/429/5xx, or an error with no usable code — is worth retrying."""
    return isinstance(code, int) and 400 <= code < 500 and code not in (408, 429)


def _error_text(error: dict) -> str:
    """OpenRouter's generic 'Provider returned error' hides the real reason
    (e.g. 'temporarily rate-limited upstream') in error.metadata.raw."""
    text = str(error.get("message"))
    raw = (error.get("metadata") or {}).get("raw") if isinstance(error.get("metadata"), dict) else None
    return f"{text}: {str(raw)[:300]}" if raw else text


def _http_error_detail(e: urllib.error.HTTPError) -> str:
    try:
        return _error_text(json.loads(e.read())["error"])
    except Exception:  # noqa: BLE001 - any unparseable body: fall back to the status line
        return str(e)
    finally:
        e.close()  # an HTTPError holds the response socket open until closed


class OpenRouter(Ollama):
    def __init__(
        self,
        api_key: str,
        base: str = DEFAULT_BASE,
        timeout: int = 900,
        max_tokens: int = 4096,
        budget_usd: float | None = None,
        usage_log: Path | None = None,
        backoff_s: float = 5,
    ):
        super().__init__(base, timeout)
        self.api_key = api_key
        self.max_tokens = max_tokens  # runaway guard; reasoning tokens count against it
        self.budget_usd = budget_usd
        self.usage_log = Path(usage_log) if usage_log else None
        self.backoff_s = backoff_s
        self.spent_usd = self._load_spent()

    def _load_spent(self) -> float:
        """Cumulative spend so far: the cap is per run directory, not per process."""
        if not self.usage_log or not self.usage_log.exists():
            return 0.0
        total = 0.0
        for line in self.usage_log.read_text().splitlines():
            try:
                total += float(json.loads(line).get("cost") or 0)
            except (json.JSONDecodeError, ValueError, AttributeError):
                continue  # torn line after a crash: skip, like the ledger does
        return total

    def _record(self, model: str, usage: dict):
        cost = float(usage.get("cost") or 0)
        self.spent_usd += cost
        if self.usage_log:
            entry = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "model": model,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
                "cost": cost,
            }
            with self.usage_log.open("a") as f:
                f.write(json.dumps(entry) + "\n")

    def _post_json(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
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
        num_ctx: int = 8192,  # accepted for interface parity; hosted models fix their own context
        retries: int = 3,
        think: bool | None = None,
    ) -> dict:
        if self.budget_usd is not None and self.spent_usd >= self.budget_usd:
            raise BudgetExceeded(f"budget ${self.budget_usd:.2f} reached (${self.spent_usd:.2f} spent)")

        payload = {
            "model": model,
            "messages": _to_openai_messages(messages),
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if think is False:
            payload["reasoning"] = {"effort": "none"}  # some models make reasoning mandatory
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "strict": False, "schema": schema},
            }
        if tools is not None:
            payload["tools"] = tools
        if schema is not None or tools is not None:
            payload["provider"] = {"require_parameters": True}  # only route to endpoints that honor them

        last = None
        for attempt in range(retries):
            try:
                data = self._post_json(payload)
            except urllib.error.HTTPError as e:
                detail = _http_error_detail(e)
                if _permanent(e.code):
                    raise OllamaError(f"OpenRouter HTTP {e.code}: {detail}") from e
                last = detail
            except (OSError, http.client.HTTPException, ValueError) as e:
                # URLError, timeouts, resets, dropped connections, non-JSON bodies:
                # all transient. Anything raw escaping here would become a placeholder scene.
                last = f"{type(e).__name__}: {e}"
            else:
                if not isinstance(data, dict):
                    last = "malformed response (not a JSON object)"
                else:
                    error = data.get("error")
                    if error and not isinstance(error, dict):
                        error = {"message": str(error)}
                    if not error and data.get("choices"):
                        return self._accept(model, data)
                    error = error or {"message": "response had no choices"}
                    if _permanent(error.get("code")):
                        raise OllamaError(f"OpenRouter error: {_error_text(error)}")
                    last = _error_text(error)
            if attempt < retries - 1:
                time.sleep(min(2**attempt * self.backoff_s, 60))
        raise OllamaError(f"OpenRouter request failed after {retries} attempts: {last}")

    def _accept(self, model: str, data: dict) -> dict:
        self._record(model, data.get("usage") or {})  # paid for either way
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            # cut off at the cap: truncated JSON / a scene stopping mid-sentence must never
            # be accepted as complete (an outline once "succeeded" ending mid-word)
            raise OllamaError(
                f"reply cut off at max_output_tokens={self.max_tokens} (finish_reason=length); "
                "raise max_output_tokens, shorten the ask, or disable reasoning"
            )
        message = choice["message"]
        result = {"role": "assistant", "content": message.get("content") or ""}
        if message.get("tool_calls"):
            result["tool_calls"] = message["tool_calls"]
        return result
