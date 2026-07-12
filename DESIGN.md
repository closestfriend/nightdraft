# wer-book-harness — Design

*Converged 2026-07-11 (Hunter + Claude), from a thread started 2026-06-28 in the philm session.
This doc is the cross-session anchor: read this before touching the code.*

## Goal

Fire-and-forget novel generation on idle hardware. Give the harness a premise; it
generates its own outline and grinds scene-by-scene, overnight after overnight, until a
complete, **coherent** draft exists. Coherence is the deliverable — the harness manages
memory outside the context window. It's a draft engine, not a finished-novel button.

Target host is `wer` (2020 Intel i9 iMac, 16 GB, CPU-only Ollama) but the harness is
**host-agnostic**: it talks to any Ollama endpoint (`OLLAMA_HOST`). Develop and test on
the M4; deploy to wer when LAN access exists again.

## Architecture: deterministic pipeline, bounded agency inside stages

Control flow is code, not model judgment (fire-and-forget + 2am + nobody watching =
no free-roaming agent). But stages use **schema-enforced tool calls** for all structured
output — July 2026 small models do this reliably (Qwen3.5 4B: 97.5% on tool-call evals)
— and the consistency stage gets a **capped tool-loop** (`search_manuscript`,
`read_bible_entry`, max N calls) so the model can retrieve before it judges.

### Two-model split (both co-resident in 16 GB)

| Role | Model (initial pick) | Why |
|---|---|---|
| Glue: outline, extraction, summaries, checks | `qwen3.5:4b` (~3.4 GB) | near-perfect tool calls, ~2× CPU speed |
| Prose: the actual scenes | 7–8B instruct, Q4 (~5 GB) | 4B prose is too thin; benchmark 2–3 candidates |

Rules learned the hard way (June image-renamer session, re-confirmed 2026-07-12):
thinking variants stall batch work — `qwen3.5:4b` spent **2m16s** of hidden reasoning
on a one-word schema answer; with `think:false` the same call took **4s**. Unlike
June's Ollama 0.30.x, the toggle is honored now, so the harness sends it on every
glue call (`glue_think` config key; set to `null` for non-thinking glue models,
since Ollama rejects the key on models without the capability). Also: benchmark
through Ollama itself, not leaderboards (chat-template mismatches sank purpose-built
tool models); **never run concurrent jobs** on the box; measure tok/s on the actual
host before committing. `brie7b` is a drop-in prose candidate for a later run.

### State: plain files, one run = one directory

```
runs/<book-slug>/
  premise.md          # the seed (only human-authored artifact)
  outline.json        # acts → chapters → scene beats; generated, then load-bearing
  bible.json          # living canon: characters, world rules, timeline, open threads
  summary.md          # rolling compressed memory of everything prior
  manuscript/         # ch01.md … the prose
  ledger.jsonl        # append-only log of every step: inputs-hash, output, timing, verdicts
```

Everything inspectable/editable in the morning. Editing `bible.json` or `outline.json`
between nights is supported human steering, not a violation.

### The per-scene loop

```
assemble context ──► generate prose ──► extract canon ──► consistency check ──► accept/revise ──► checkpoint
(bible slice +      (prose model)      (glue model,      (glue model, capped     (1 retry max,     (write files +
 rolling summary +                      tool schema →     tool-loop over          then accept       ledger append)
 prev scene tail +                      bible updates)    manuscript + bible)     with flag)
 this beat)
```

- **Scene-level check** (every scene): does this scene contradict the bible? Violations
  → one bounded revise pass; if still failing, accept + flag in ledger (draft engine —
  a flagged wart beats a stalled run).
- **Outline-level drift check** (every N chapters, N=3 default): glue model compares
  rolling summary against outline; may make bounded course-corrections to *remaining*
  beats only. All self-edits logged to the ledger for morning audit.

### Failure semantics (non-negotiables)

- Any single call failing must never kill the run: retry w/ backoff → skip + flag → move on.
- Crash/power-loss = **resume, not restart**: ledger + checkpoint files are the truth;
  re-running the CLI continues from the last accepted scene.
- Timeouts generous (CPU inference: minutes/scene is normal, not a hang).
- Validation on every structured output; one salvage attempt; then flag + continue.

## CLI shape

```
book init <slug> --premise premise.md    # scaffold a run
book plan <slug>                         # premise → outline.json + initial bible.json
book run <slug> [--until-dawn|--scenes N]  # the overnight loop
book status <slug>                       # progress, flags, drift verdicts
book compile <slug>                      # manuscript/*.md → single book.md/epub
```

Python 3.11+, stdlib + `requests` only. No framework. Runs anywhere; launchd plist
(+ `caffeinate`) ships in `deploy/` for wer.

## wer deployment checklist (next office visit)

1. Install Ollama; `ollama pull` the two models.
2. Install Tailscale → never LAN-blocked again.
3. System Settings: auto-restart after power failure; wake for network access; never sleep (or launchd + caffeinate).
4. Benchmark: tok/s both models, one full scene loop timed end-to-end.
5. Kick off first overnight; check ledger over coffee.

## Build order

1. **Skeleton + state** — run dir scaffolding, ledger append/replay, resume logic. (Testable with a mock model.)
2. **Glue calls** — Ollama client, tool-schema enforcement, salvage/retry wrapper.
3. **`book plan`** — premise → outline + seed bible.
4. **Scene loop** — assemble/generate/extract/check/checkpoint, scene-level only.
5. **Drift check + revise pass.**
6. **`book compile` + `book status`.**
7. **M4 end-to-end**: generate a short novella (3 chapters) as the acceptance test.

Ship after step 7; wer deployment is a config change, not a code change.

## Explicitly out of scope (v1)

Retrieval/embeddings over the manuscript (the capped tool-loop's grep-style search is
v1's retrieval; add embeddings only if morning audits show continuity misses), cloud
prose/editor passes, multi-book queueing, any UI beyond the CLI.
