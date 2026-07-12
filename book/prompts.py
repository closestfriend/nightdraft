"""All prompts and schemas in one place — the harness's actual craft lives here."""

# ---------- planning ----------

OUTLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "logline": {"type": "string"},
        "acts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "chapters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "scenes": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {"beat": {"type": "string"}},
                                        "required": ["beat"],
                                    },
                                },
                            },
                            "required": ["title", "scenes"],
                        },
                    },
                },
                "required": ["title", "chapters"],
            },
        },
    },
    "required": ["title", "logline", "acts"],
}

def outline_prompt(premise: str, chapters: int, scenes_per_chapter: int) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a story architect. You design novel outlines with strong "
                "causality: every scene beat states what HAPPENS and what CHANGES. "
                "Beats are 1-3 sentences, concrete, and name the characters involved."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Design a novel outline from this premise.\n\n"
                f"PREMISE:\n{premise}\n\n"
                f"Structure: 3 acts, ~{chapters} chapters total, "
                f"~{scenes_per_chapter} scenes per chapter. Escalate stakes across acts; "
                f"plant at least two setups early that pay off in act 3."
            ),
        },
    ]

BIBLE_SEED_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section": {"type": "string", "enum": ["characters", "world", "threads"]},
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["section", "key", "value"],
            },
        }
    },
    "required": ["facts"],
}

def bible_seed_prompt(premise: str, outline_json: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You extract a story bible from a premise and outline. Facts must be "
                "canon-worthy: character traits/voices/goals, world rules, and open "
                "plot threads. key = the character/rule/thread name; value = one "
                "declarative sentence of canon."
            ),
        },
        {
            "role": "user",
            "content": f"PREMISE:\n{premise}\n\nOUTLINE:\n{outline_json}\n\nExtract the seed bible facts.",
        },
    ]

# ---------- prose ----------

PROSE_SYSTEM = (
    "You are a novelist writing one scene of a longer book. Write vivid, concrete "
    "prose in close third person, past tense. Show character through action and "
    "dialogue. Do NOT summarize, do NOT write chapter headings, do NOT conclude the "
    "book — write only this scene, ending on a note that pulls the reader forward. "
    "Stay strictly consistent with the CANON provided."
)

def prose_prompt(
    beat: str,
    bible_slice: str,
    rolling_summary: str,
    prev_tail: str,
    scene_words: int,
    revision_notes: str = "",
) -> list[dict]:
    parts = [f"CANON (do not contradict):\n{bible_slice}"]
    if rolling_summary:
        parts.append(f"THE STORY SO FAR:\n{rolling_summary}")
    if prev_tail:
        parts.append(f"THE PREVIOUS SCENE ENDED:\n…{prev_tail}")
    parts.append(f"WRITE THIS SCENE (~{scene_words} words):\n{beat}")
    if revision_notes:
        parts.append(
            "REVISION PASS — your previous draft violated canon. Fix these issues "
            f"while keeping everything that worked:\n{revision_notes}"
        )
    return [
        {"role": "system", "content": PROSE_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]

# ---------- extraction ----------

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "scene_summary": {"type": "string"},
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": ["characters", "world", "timeline", "threads"],
                    },
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["section", "key", "value"],
            },
        },
        "threads_closed": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["scene_summary", "facts", "threads_closed"],
}

def extract_prompt(scene_prose: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You maintain a novel's story bible. From the scene, extract NEW canon "
                "only: character facts revealed, world rules established, timeline "
                "events, threads opened. List any existing threads this scene resolves "
                "in threads_closed. scene_summary = 1-2 sentences of what happened and "
                "what changed. For timeline facts, key = a short event label."
            ),
        },
        {"role": "user", "content": f"SCENE:\n{scene_prose}"},
    ]

# ---------- consistency check (capped tool loop) ----------

CHECK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_manuscript",
            "description": "Search all previously written scenes for a phrase/name; returns matching lines.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_bible_entry",
            "description": "Read the canon notes for one character/world-rule/thread by name.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    },
]

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "consistent": {"type": "boolean"},
        "violations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "what": {"type": "string"},
                    "severity": {"type": "string", "enum": ["minor", "major"]},
                },
                "required": ["what", "severity"],
            },
        },
    },
    "required": ["consistent", "violations"],
}

CHECK_SYSTEM = (
    "You are a continuity editor. Check the new scene against the story canon. "
    "You may call tools to verify suspicions against earlier scenes and the bible. "
    "Major = contradicts established canon (a fact, a death, a rule, the timeline). "
    "Minor = tonal or detail wobble. Be skeptical but not pedantic: only report real "
    "contradictions, not stylistic choices."
)

def check_prompt(scene_prose: str, bible_slice: str) -> list[dict]:
    return [
        {"role": "system", "content": CHECK_SYSTEM},
        {
            "role": "user",
            "content": (
                f"CANON:\n{bible_slice}\n\nNEW SCENE:\n{scene_prose}\n\n"
                "Investigate anything suspicious, then deliver your verdict."
            ),
        },
    ]

VERDICT_REQUEST = (
    "Based on your investigation, deliver the final verdict now as JSON "
    "(consistent + violations)."
)

# ---------- summary compression & drift ----------

def compress_prompt(chapter_bullets: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "Compress scene-by-scene bullets into one dense paragraph (3-5 "
                "sentences) preserving: who did what, what changed, what remains open. "
                "This paragraph becomes the book's only memory of the chapter."
            ),
        },
        {"role": "user", "content": chapter_bullets},
    ]

DRIFT_SCHEMA = {
    "type": "object",
    "properties": {
        "on_course": {"type": "boolean"},
        "diagnosis": {"type": "string"},
        "revised_beats": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene_id": {"type": "string"},
                    "new_beat": {"type": "string"},
                },
                "required": ["scene_id", "new_beat"],
            },
        },
    },
    "required": ["on_course", "diagnosis", "revised_beats"],
}

def drift_prompt(outline_json: str, rolling_summary: str, remaining_ids: list[str]) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are the story's showrunner doing a mid-season check. Compare what "
                "has actually been written (the summary) against the plan (the "
                "outline). The story may have improvised. If the improvisations broke "
                "the road to the planned ending, revise UPCOMING beats to reconnect — "
                "bend the plan toward what the story became, don't fight it. You may "
                f"ONLY revise these upcoming scenes: {', '.join(remaining_ids)}. "
                "If the story is on course, revise nothing."
            ),
        },
        {
            "role": "user",
            "content": f"OUTLINE:\n{outline_json}\n\nWHAT WAS ACTUALLY WRITTEN:\n{rolling_summary}",
        },
    ]
