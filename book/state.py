"""Run state: one directory per book, plain files, append-only ledger.

The ledger is the source of truth for progress. Crash/power-loss recovery is
"replay the ledger", never "start over". Everything else (bible, summary,
manuscript) is a checkpoint file that the morning human may freely edit.
"""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = {
    "host": None,  # None -> $OLLAMA_HOST or localhost
    "glue_model": "qwen3.5:4b",
    "prose_model": "brie7b:latest",
    "scene_words": 900,
    "chapters": 12,
    "scenes_per_chapter": 4,
    "drift_every_chapters": 3,
    "tool_call_cap": 4,
    "num_ctx": 8192,
    "prose_temperature": 0.9,
    "glue_temperature": 0.2,
    "request_timeout_s": 900,
}


@dataclass
class Beat:
    """One scene's worth of planned story, addressable as e.g. c03s02."""

    chapter: int
    scene: int
    beat: str
    chapter_title: str

    @property
    def id(self) -> str:
        return f"c{self.chapter:02d}s{self.scene:02d}"


class Run:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.slug = self.root.name
        self.manuscript_dir = self.root / "manuscript"
        self.ledger_path = self.root / "ledger.jsonl"
        self.outline_path = self.root / "outline.json"
        self.bible_path = self.root / "bible.json"
        self.summary_path = self.root / "summary.md"
        self.premise_path = self.root / "premise.md"
        self.config_path = self.root / "config.json"

    # ---------- creation / loading ----------

    @classmethod
    def create(cls, runs_dir: Path, slug: str, premise_text: str, overrides: dict | None = None):
        root = Path(runs_dir) / slug
        if root.exists():
            raise FileExistsError(f"run already exists: {root}")
        run = cls(root)
        run.manuscript_dir.mkdir(parents=True)
        run.premise_path.write_text(premise_text)
        config = dict(DEFAULT_CONFIG)
        config.update(overrides or {})
        run.config_path.write_text(json.dumps(config, indent=2))
        run.summary_path.write_text("")
        run.log("run_created", {"slug": slug})
        return run

    @classmethod
    def open(cls, runs_dir: Path, slug: str):
        root = Path(runs_dir) / slug
        if not root.is_dir():
            raise FileNotFoundError(f"no such run: {root}")
        return cls(root)

    @property
    def config(self) -> dict:
        cfg = dict(DEFAULT_CONFIG)
        if self.config_path.exists():
            cfg.update(json.loads(self.config_path.read_text()))
        return cfg

    @property
    def premise(self) -> str:
        return self.premise_path.read_text()

    # ---------- outline / beats ----------

    def outline(self) -> dict | None:
        if not self.outline_path.exists():
            return None
        return json.loads(self.outline_path.read_text())

    def write_outline(self, outline: dict, reason: str):
        if self.outline_path.exists():
            backup = self.outline_path.with_suffix(f".{int(time.time())}.bak.json")
            backup.write_text(self.outline_path.read_text())
        self.outline_path.write_text(json.dumps(outline, indent=2))
        self.log("outline_written", {"reason": reason})

    def beats(self) -> list[Beat]:
        outline = self.outline()
        if outline is None:
            return []
        out: list[Beat] = []
        chapter_n = 0
        for act in outline.get("acts", []):
            for chapter in act.get("chapters", []):
                chapter_n += 1
                for i, scene in enumerate(chapter.get("scenes", []), start=1):
                    out.append(
                        Beat(
                            chapter=chapter_n,
                            scene=i,
                            beat=scene["beat"],
                            chapter_title=chapter.get("title", f"Chapter {chapter_n}"),
                        )
                    )
        return out

    # ---------- ledger ----------

    def log(self, step: str, data: dict | None = None, **kw):
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "step": step}
        entry.update(data or {})
        entry.update(kw)
        with self.ledger_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def ledger(self) -> list[dict]:
        if not self.ledger_path.exists():
            return []
        entries = []
        for line in self.ledger_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn write from a crash; the entry is lost, the run is not
        return entries

    def accepted_scene_ids(self) -> set[str]:
        return {e["scene_id"] for e in self.ledger() if e["step"] == "scene_accepted"}

    def next_beat(self) -> Beat | None:
        done = self.accepted_scene_ids()
        for beat in self.beats():
            if beat.id not in done:
                return beat
        return None

    def flags(self) -> list[dict]:
        return [e for e in self.ledger() if e.get("flag")]

    # ---------- bible ----------

    def bible(self) -> dict:
        if not self.bible_path.exists():
            return {"characters": {}, "world": {}, "timeline": [], "threads": {}}
        return json.loads(self.bible_path.read_text())

    def write_bible(self, bible: dict):
        self.bible_path.write_text(json.dumps(bible, indent=2))

    def merge_facts(self, facts: list[dict], scene_id: str) -> int:
        """Merge extracted facts into the bible with provenance. Returns count merged."""
        bible = self.bible()
        merged = 0
        for fact in facts:
            section = fact.get("section")
            key = (fact.get("key") or "").strip()
            value = (fact.get("value") or "").strip()
            if not key or not value:
                continue
            if section == "timeline":
                bible["timeline"].append({"event": value, "scene": scene_id})
                merged += 1
            elif section in ("characters", "world", "threads"):
                entry = bible[section].setdefault(key, {"notes": [], "scenes": []})
                if value not in entry["notes"]:
                    entry["notes"].append(value)
                    merged += 1
                if scene_id not in entry["scenes"]:
                    entry["scenes"].append(scene_id)
        self.write_bible(bible)
        return merged

    def close_thread(self, key: str, scene_id: str):
        bible = self.bible()
        thread = bible["threads"].get(key)
        if thread is not None:
            thread["closed_in"] = scene_id
            self.write_bible(bible)

    # ---------- rolling summary ----------

    def summary(self) -> str:
        return self.summary_path.read_text() if self.summary_path.exists() else ""

    def append_scene_summary(self, beat: Beat, line: str):
        text = self.summary()
        header = f"## Chapter {beat.chapter}: {beat.chapter_title}"
        if header not in text:
            text += f"\n\n{header}\n"
        text += f"- ({beat.id}) {line.strip()}\n"
        self.summary_path.write_text(text)

    def replace_chapter_summary(self, chapter: int, chapter_title: str, paragraph: str):
        """Compress a chapter's scene bullets into one paragraph (end-of-chapter)."""
        text = self.summary()
        header = f"## Chapter {chapter}: {chapter_title}"
        pattern = re.escape(header) + r"\n(?:- .*\n?)*"
        replacement = f"{header}\n{paragraph.strip()}\n"
        new_text, n = re.subn(pattern, replacement, text)
        self.summary_path.write_text(new_text if n else text + f"\n\n{replacement}")

    # ---------- manuscript ----------

    def scene_path(self, beat: Beat) -> Path:
        return self.manuscript_dir / f"{beat.id}.md"

    def write_scene(self, beat: Beat, prose: str):
        self.scene_path(beat).write_text(prose.strip() + "\n")

    def scene_files(self) -> list[Path]:
        return sorted(self.manuscript_dir.glob("c*s*.md"))

    def previous_scene_tail(self, beat: Beat, words: int = 400) -> str:
        """Tail of the most recent scene before this beat, for prose continuity."""
        earlier = [p for p in self.scene_files() if p.stem < beat.id]
        if not earlier:
            return ""
        tail_words = earlier[-1].read_text().split()
        return " ".join(tail_words[-words:])

    def search_manuscript(self, query: str, max_hits: int = 6) -> list[dict]:
        """Case-insensitive substring search; the v1 retrieval story."""
        hits = []
        needle = query.lower().strip()
        if not needle:
            return hits
        for path in self.scene_files():
            for i, line in enumerate(path.read_text().splitlines()):
                if needle in line.lower():
                    hits.append({"scene": path.stem, "line": line.strip()[:300]})
                    if len(hits) >= max_hits:
                        return hits
        return hits
