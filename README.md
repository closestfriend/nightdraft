# nightdraft

Fire-and-forget novel generation on idle hardware. Give it a premise; it plans an
outline, then grinds scene-by-scene — maintaining a story bible, a rolling summary,
and consistency/drift checks — until a coherent draft exists.

**Read `DESIGN.md` first.** It's the cross-session anchor: architecture, failure
posture, model rules, and the wer deployment checklist all live there.

## Quickstart

```bash
echo "Your premise here..." > premise.md
python3 -m book init my-novel --premise premise.md
python3 -m book plan my-novel        # premise -> outline + seed bible
python3 -m book run my-novel         # the overnight loop (resumes automatically)
python3 -m book status my-novel     # progress + flags for morning review
python3 -m book compile my-novel    # -> runs/my-novel/book.md
```

Point at a remote Ollama (e.g. wer) with `OLLAMA_HOST=wer:11434` or the `"host"`
key in the run's `config.json`.

Everything about a run lives in `runs/<slug>/` as plain files — edit `bible.json`
or `outline.json` between nights to steer; the ledger (`ledger.jsonl`) is
append-only truth and makes every run resumable after crash or power loss.

## Tests

```bash
python3 -m unittest discover -s tests
```

No dependencies: Python 3.10+ stdlib only.
