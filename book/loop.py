"""The overnight loop: assemble -> generate -> extract -> check -> checkpoint.

Failure posture everywhere: a scene that can't be perfected gets accepted with
a flag in the ledger. A flagged wart beats a stalled run at 2am.
"""

import json

from . import prompts
from .ollama import Ollama, OllamaError, extract_json
from .state import Beat, Run


def bible_slice(run: Run, beat: Beat, max_chars: int = 4000) -> str:
    """Canon relevant to this beat: named characters first, world rules,
    open threads. Crude name-matching is deliberate v1 (see DESIGN.md)."""
    bible = run.bible()
    lines = []
    beat_lower = beat.beat.lower()
    for name, entry in bible["characters"].items():
        marker = "* " if name.lower() in beat_lower else ""
        lines.append(f"{marker}{name}: " + " ".join(entry["notes"]))
    for rule, entry in bible["world"].items():
        lines.append(f"[world] {rule}: " + " ".join(entry["notes"]))
    for thread, entry in bible["threads"].items():
        status = "CLOSED" if entry.get("closed_in") else "OPEN"
        lines.append(f"[thread/{status}] {thread}: " + " ".join(entry["notes"]))
    if bible["timeline"]:
        recent = bible["timeline"][-8:]
        lines.append("[timeline] " + " → ".join(e["event"] for e in recent))
    # characters in this scene sort to the top; then trim to budget
    lines.sort(key=lambda l: not l.startswith("*"))
    out = []
    total = 0
    for line in lines:
        if total + len(line) > max_chars:
            break
        out.append(line)
        total += len(line)
    return "\n".join(out) or "(no canon yet)"


def consistency_check(run: Run, client: Ollama, scene_prose: str, slice_text: str) -> dict:
    """Glue model investigates with tools (capped), then delivers a verdict."""
    cfg = run.config
    messages = prompts.check_prompt(scene_prose, slice_text)
    calls_used = 0
    while calls_used < cfg["tool_call_cap"]:
        msg = client.chat(
            cfg["glue_model"],
            messages,
            tools=prompts.CHECK_TOOLS,
            temperature=cfg["glue_temperature"],
            think=cfg.get("glue_think"),
            num_ctx=cfg["num_ctx"],
        )
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            break
        messages.append(msg)
        for call in tool_calls:
            calls_used += 1
            fn = call.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                args = extract_json(args) or {}
            if fn.get("name") == "search_manuscript":
                result = run.search_manuscript(args.get("query", ""))
            elif fn.get("name") == "read_bible_entry":
                key = args.get("key", "")
                bible = run.bible()
                result = next(
                    (sec[key] for sec in (bible["characters"], bible["world"], bible["threads"]) if key in sec),
                    {"error": f"no canon entry named {key!r}"},
                )
            else:
                result = {"error": "unknown tool"}
            messages.append({"role": "tool", "content": json.dumps(result)})
    # final verdict, schema-constrained (tools and format don't mix in one call)
    messages.append({"role": "user", "content": prompts.VERDICT_REQUEST})
    return client.chat_json(
        cfg["glue_model"],
        messages,
        schema=prompts.VERDICT_SCHEMA,
        temperature=cfg["glue_temperature"],
        think=cfg.get("glue_think"),
        num_ctx=cfg["num_ctx"],
    )


def write_scene(run: Run, client: Ollama, beat: Beat) -> dict:
    """One full scene cycle. Returns the ledger data for scene_accepted."""
    cfg = run.config
    slice_text = bible_slice(run, beat)
    flags = []

    def generate(revision_notes: str = "") -> str:
        msg = client.chat(
            cfg["prose_model"],
            prompts.prose_prompt(
                beat.beat,
                slice_text,
                run.summary(),
                run.previous_scene_tail(beat),
                cfg["scene_words"],
                revision_notes,
            ),
            temperature=cfg["prose_temperature"],
            num_ctx=cfg["num_ctx"],
        )
        return (msg.get("content") or "").strip()

    prose = generate()
    if len(prose.split()) < cfg["scene_words"] // 4:
        flags.append("short_scene")

    # consistency check + at most one revision pass
    verdict = {"consistent": True, "violations": []}
    try:
        verdict = consistency_check(run, client, prose, slice_text)
        majors = [v for v in verdict["violations"] if v["severity"] == "major"]
        if majors:
            notes = "\n".join(f"- {v['what']}" for v in majors)
            run.log("scene_revising", {"scene_id": beat.id, "violations": notes})
            prose = generate(revision_notes=notes) or prose
            verdict = consistency_check(run, client, prose, slice_text)
            if any(v["severity"] == "major" for v in verdict["violations"]):
                flags.append("unresolved_canon_violation")
    except OllamaError as e:
        flags.append(f"check_failed:{e}")

    # extract canon (a failure here flags rather than blocks)
    scene_summary = ""
    try:
        extraction = client.chat_json(
            cfg["glue_model"],
            prompts.extract_prompt(prose),
            schema=prompts.EXTRACT_SCHEMA,
            temperature=cfg["glue_temperature"],
            think=cfg.get("glue_think"),
            num_ctx=cfg["num_ctx"],
        )
        run.merge_facts(extraction.get("facts", []), beat.id)
        for thread in extraction.get("threads_closed", []):
            run.close_thread(thread, beat.id)
        scene_summary = extraction.get("scene_summary", "")
    except OllamaError as e:
        flags.append(f"extract_failed:{e}")

    run.write_scene(beat, prose)
    run.append_scene_summary(beat, scene_summary or beat.beat)
    return {
        "scene_id": beat.id,
        "words": len(prose.split()),
        "violations": verdict["violations"],
        "flag": bool(flags),
        "flags": flags,
    }


def end_of_chapter(run: Run, client: Ollama, beat: Beat):
    """Compress the finished chapter's bullets; maybe run the drift check."""
    cfg = run.config
    header = f"## Chapter {beat.chapter}: {beat.chapter_title}"
    text = run.summary()
    start = text.find(header)
    bullets = text[start:].split("\n\n")[0] if start != -1 else ""
    try:
        msg = client.chat(
            cfg["glue_model"],
            prompts.compress_prompt(bullets),
            temperature=cfg["glue_temperature"],
            think=cfg.get("glue_think"),
            num_ctx=cfg["num_ctx"],
        )
        paragraph = (msg.get("content") or "").strip()
        if paragraph:
            run.replace_chapter_summary(beat.chapter, beat.chapter_title, paragraph)
    except OllamaError:
        run.log("compress_failed", {"chapter": beat.chapter, "flag": True})

    if beat.chapter % cfg["drift_every_chapters"] == 0:
        drift_check(run, client, current_chapter=beat.chapter)


def drift_check(run: Run, client: Ollama, current_chapter: int):
    cfg = run.config
    remaining = [b for b in run.beats() if b.chapter > current_chapter]
    if not remaining:
        return
    remaining_ids = [b.id for b in remaining]
    try:
        result = client.chat_json(
            cfg["glue_model"],
            prompts.drift_prompt(
                json.dumps(run.outline(), indent=2), run.summary(), remaining_ids
            ),
            schema=prompts.DRIFT_SCHEMA,
            temperature=cfg["glue_temperature"],
            think=cfg.get("glue_think"),
            num_ctx=cfg["num_ctx"],
        )
    except OllamaError as e:
        run.log("drift_check_failed", {"chapter": current_chapter, "error": str(e), "flag": True})
        return

    applied = []
    if not result["on_course"] and result["revised_beats"]:
        outline = run.outline()
        by_id = {b.id: b for b in remaining}
        chapter_n = 0
        for act in outline["acts"]:
            for chapter in act["chapters"]:
                chapter_n += 1
                for i, scene in enumerate(chapter["scenes"], start=1):
                    sid = f"c{chapter_n:02d}s{i:02d}"
                    for rev in result["revised_beats"]:
                        if rev["scene_id"] == sid and sid in by_id:  # upcoming only
                            scene["beat"] = rev["new_beat"]
                            applied.append(sid)
        if applied:
            run.write_outline(outline, reason=f"drift correction after ch{current_chapter}")
    run.log(
        "drift_check",
        {
            "chapter": current_chapter,
            "on_course": result["on_course"],
            "diagnosis": result["diagnosis"],
            "beats_revised": applied,
            "flag": not result["on_course"],
        },
    )


def run_loop(run: Run, client: Ollama | None = None, max_scenes: int | None = None) -> int:
    """The overnight entry point. Resumes wherever the ledger says we are.
    Returns the number of scenes written this session."""
    cfg = run.config
    client = client or Ollama(cfg["host"], timeout=cfg["request_timeout_s"])
    if not run.beats():
        raise RuntimeError("no outline — run `book plan` first")

    written = 0
    while max_scenes is None or written < max_scenes:
        beat = run.next_beat()
        if beat is None:
            run.log("draft_complete", {"scenes": len(run.beats())})
            return written
        try:
            data = write_scene(run, client, beat)
        except OllamaError as e:
            # scene-level catch: skip + flag + move on, never die
            run.log("scene_failed", {"scene_id": beat.id, "error": str(e), "flag": True})
            run.write_scene(beat, f"[SCENE FAILED: {beat.beat}]")
            run.log("scene_accepted", {"scene_id": beat.id, "flag": True, "flags": ["generation_failed"]})
            written += 1
            continue
        run.log("scene_accepted", data)
        written += 1

        nxt = run.next_beat()
        if nxt is None or nxt.chapter != beat.chapter:
            end_of_chapter(run, client, beat)
    return written
