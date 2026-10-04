"""CLI: book init | plan | run | status | compile"""

import argparse
import json
import sys
from pathlib import Path

from .state import Run

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def cmd_init(args):
    premise = Path(args.premise).read_text()
    overrides = json.loads(args.config) if args.config else {}
    run = Run.create(args.runs_dir, args.slug, premise, overrides)
    print(f"created {run.root}")
    print("next: book plan " + args.slug)


def cmd_plan(args):
    from .plan import plan  # deferred: keeps status/compile usable without network

    run = Run.open(args.runs_dir, args.slug)
    outline = plan(run)
    print(f"“{outline['title']}” — {outline['logline']}")
    print(f"{len(run.beats())} scenes planned. next: book run {args.slug}")


def cmd_run(args):
    from .loop import run_loop

    run = Run.open(args.runs_dir, args.slug)
    n = run_loop(run, max_scenes=args.scenes)
    done = len(run.accepted_scene_ids())
    total = len(run.beats())
    print(f"wrote {n} scene(s); {done}/{total} complete")
    if done >= total:
        print(f"draft complete. next: book compile {args.slug}")


def cmd_status(args):
    run = Run.open(args.runs_dir, args.slug)
    beats = run.beats()
    done = run.accepted_scene_ids()
    outline = run.outline()
    title = outline.get("title", args.slug) if outline else args.slug
    print(f"“{title}” — {len(done)}/{len(beats)} scenes")
    words = sum(len(p.read_text().split()) for p in run.scene_files())
    print(f"manuscript: {words} words across {len(run.scene_files())} scene files")
    flags = run.flags()
    if flags:
        print(f"\n{len(flags)} flagged event(s) for morning review:")
        for e in flags[-15:]:
            detail = e.get("flags") or e.get("diagnosis") or e.get("error") or ""
            print(f"  {e['ts']}  {e['step']}  {e.get('scene_id', '')}  {detail}")
    else:
        print("no flags — clean run so far")


def cmd_compile(args):
    run = Run.open(args.runs_dir, args.slug)
    outline = run.outline() or {}
    parts = [f"# {outline.get('title', args.slug)}\n"]
    if outline.get("logline"):
        parts.append(f"*{outline['logline']}*\n")
    current_chapter = None
    for beat in run.beats():
        path = run.scene_path(beat)
        if not path.exists():
            continue
        if beat.chapter != current_chapter:
            current_chapter = beat.chapter
            parts.append(f"\n\n## Chapter {beat.chapter}: {beat.chapter_title}\n")
        parts.append(path.read_text().strip())
        parts.append("")
    out = run.root / "book.md"
    out.write_text("\n".join(parts))
    words = len(out.read_text().split())
    print(f"compiled {out} ({words} words)")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="book", description="nightdraft")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="scaffold a new run from a premise file")
    p.add_argument("slug")
    p.add_argument("--premise", required=True)
    p.add_argument("--config", help="JSON string of config overrides")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("plan", help="premise -> outline + seed bible")
    p.add_argument("slug")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("run", help="the overnight loop (resumes automatically)")
    p.add_argument("slug")
    p.add_argument("--scenes", type=int, default=None, help="stop after N scenes (default: run to completion)")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("status", help="progress + flags")
    p.add_argument("slug")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("compile", help="manuscript/*.md -> book.md")
    p.add_argument("slug")
    p.set_defaults(fn=cmd_compile)

    args = parser.parse_args(argv)
    try:
        args.fn(args)
    except (FileNotFoundError, FileExistsError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0
