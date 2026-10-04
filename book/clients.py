"""Client factory: the one place that decides which model backend a run talks to."""

import os

from .ollama import Ollama
from .openrouter import DEFAULT_BASE, OpenRouter
from .state import Run


def make_client(run: Run):
    cfg = run.config
    if cfg.get("provider", "ollama") == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("provider is 'openrouter' but OPENROUTER_API_KEY is not set in the environment")
        return OpenRouter(
            key,
            base=cfg.get("openrouter_base") or DEFAULT_BASE,
            timeout=cfg["request_timeout_s"],
            max_tokens=cfg.get("max_output_tokens", 4096),
            budget_usd=cfg.get("budget_usd"),
            usage_log=run.root / "usage.jsonl",
        )
    return Ollama(cfg["host"], timeout=cfg["request_timeout_s"])
