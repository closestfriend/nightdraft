"""`book plan`: premise -> outline.json + seed bible.json."""

import json

from . import prompts
from .ollama import Ollama
from .state import Run


def plan(run: Run, client: Ollama | None = None):
    cfg = run.config
    client = client or Ollama(cfg["host"], timeout=cfg["request_timeout_s"])
    glue = cfg["glue_model"]

    outline = client.chat_json(
        glue,
        prompts.outline_prompt(run.premise, cfg["chapters"], cfg["scenes_per_chapter"]),
        schema=prompts.OUTLINE_SCHEMA,
        temperature=0.7,  # planning wants some invention, unlike the other glue calls
        num_ctx=cfg["num_ctx"],
    )
    run.write_outline(outline, reason="initial plan")

    seed = client.chat_json(
        glue,
        prompts.bible_seed_prompt(run.premise, json.dumps(outline, indent=2)),
        schema=prompts.BIBLE_SEED_SCHEMA,
        temperature=cfg["glue_temperature"],
        think=cfg.get("glue_think"),
        num_ctx=cfg["num_ctx"],
    )
    merged = run.merge_facts(seed.get("facts", []), scene_id="seed")
    run.log("planned", {"beats": len(run.beats()), "seed_facts": merged, "title": outline.get("title")})
    return outline
