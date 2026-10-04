# wer deployment checklist (next office visit)

wer = 2020 Intel i9-10910 iMac, 16 GB, macOS 15.7, `ssh wer` (wer-iMac.local, LAN only).
Intel = CPU-only Ollama. That's fine: overnight text batch is exactly the workload.

## One-time setup (requires physical/LAN access)

1. **Ollama**: install from ollama.com, then
   `ollama pull qwen3.5:4b && ollama pull <prose-model>`
   (~9 GB total; wer had ~36 GB free as of 2026-07-09 — check first.)
2. **Tailscale**: install + sign in → wer stops being LAN-locked forever.
   This is the single highest-value step; do it even if nothing else gets done.
3. **System Settings → Energy**:
   - "Start up automatically after a power failure" ON
   - "Wake for network access" ON
   - Prevent sleep (or rely on the launchd job's `caffeinate`)
4. **Benchmark before committing** (June lesson: measure, don't guess):
   ```bash
   OLLAMA_HOST=wer:11434 python3 -m book run <slug> --scenes 1
   ```
   Time one full scene cycle end-to-end. Decide chapter targets from real numbers.
5. **Serve on the LAN**: `launchctl setenv OLLAMA_HOST 0.0.0.0` on wer (or run the
   harness ON wer itself — it's stdlib-only, just clone the repo there).

## Nightly job (harness runs on wer)

`deploy/com.hnsk.bookharness.plist` — install with:
```bash
scp deploy/com.hnsk.bookharness.plist wer:~/Library/LaunchAgents/
ssh wer 'launchctl load ~/Library/LaunchAgents/com.hnsk.bookharness.plist'
```
Edit the slug in the plist's ProgramArguments first. It fires at 23:30 nightly,
wrapped in caffeinate; the run resumes from the ledger each night automatically,
and exits when the draft completes.

## Morning routine (from anywhere, once Tailscale is up)

```bash
ssh wer 'cd nightdraft && python3 -m book status <slug>'
```
Read the flags. Edit `bible.json` / upcoming `outline.json` beats if steering is
needed. The drift check logs its own course-corrections to the ledger — audit them.
