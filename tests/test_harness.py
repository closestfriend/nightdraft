"""Harness tests with a mock model — the parts that must never break:
ledger/resume semantics, JSON salvage, and the full scene loop."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from book.loop import run_loop, write_scene  # noqa: E402
from book.ollama import OllamaError, extract_json  # noqa: E402
from book.state import Run  # noqa: E402

OUTLINE = {
    "title": "Test Novel",
    "logline": "A test.",
    "acts": [
        {
            "title": "Act 1",
            "chapters": [
                {"title": "Beginnings", "scenes": [{"beat": "Mara finds the door."}, {"beat": "Mara opens the door."}]},
                {"title": "Middles", "scenes": [{"beat": "Beyond the door."}]},
            ],
        }
    ],
}


class MockOllama:
    """Scriptable stand-in for the Ollama client (duck-typed interface)."""

    def __init__(self):
        self.calls = []
        self.fail_models = set()

    def chat(self, model, messages, schema=None, tools=None, **kw):
        self.calls.append({"model": model, "tools": bool(tools), "schema": bool(schema)})
        if model in self.fail_models:
            raise OllamaError("mock failure")
        if tools:  # consistency investigation: one tool round, then done
            if not any(m.get("role") == "tool" for m in messages):
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "search_manuscript", "arguments": {"query": "door"}}}
                    ],
                }
            return {"role": "assistant", "content": "Looks fine."}
        if "Compress" in messages[0].get("content", ""):
            return {"role": "assistant", "content": "Compressed chapter: Mara progressed."}
        return {"role": "assistant", "content": "The prose of the scene. " * 60}

    def chat_json(self, model, messages, schema, **kw):
        self.calls.append({"model": model, "json": True})
        if model in self.fail_models:
            raise OllamaError("mock failure")
        props = schema.get("properties", {})
        if "consistent" in props:
            return {"consistent": True, "violations": []}
        if "scene_summary" in props:
            return {
                "scene_summary": "Mara progressed.",
                "facts": [{"section": "characters", "key": "Mara", "value": "Is curious."}],
                "threads_closed": [],
            }
        if "on_course" in props:
            return {"on_course": True, "diagnosis": "fine", "revised_beats": []}
        return {}


def make_run(tmp):
    run = Run.create(Path(tmp), "test-book", "A premise.")
    run.write_outline(OUTLINE, reason="test")
    return run


class TestSalvage(unittest.TestCase):
    def test_bare(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced(self):
        self.assertEqual(extract_json('Sure!\n```json\n{"a": 1}\n```'), {"a": 1})

    def test_embedded_nested(self):
        self.assertEqual(extract_json('text {"a": {"b": 2}} trailing'), {"a": {"b": 2}})

    def test_hopeless(self):
        self.assertIsNone(extract_json("no json here"))


class TestStateAndResume(unittest.TestCase):
    def test_beats_flatten_and_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(tmp)
            beats = run.beats()
            self.assertEqual([b.id for b in beats], ["c01s01", "c01s02", "c02s01"])

    def test_resume_skips_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(tmp)
            run.log("scene_accepted", {"scene_id": "c01s01"})
            self.assertEqual(run.next_beat().id, "c01s02")
            # reopening from disk sees the same state (crash recovery)
            run2 = Run.open(Path(tmp), "test-book")
            self.assertEqual(run2.next_beat().id, "c01s02")

    def test_torn_ledger_line_is_survivable(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(tmp)
            run.log("scene_accepted", {"scene_id": "c01s01"})
            with run.ledger_path.open("a") as f:
                f.write('{"step": "scene_acc')  # power died mid-write
            self.assertEqual(run.accepted_scene_ids(), {"c01s01"})

    def test_bible_merge_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(tmp)
            n = run.merge_facts(
                [
                    {"section": "characters", "key": "Mara", "value": "Curious."},
                    {"section": "characters", "key": "Mara", "value": "Curious."},  # dup
                    {"section": "timeline", "key": "found", "value": "Mara found the door."},
                ],
                "c01s01",
            )
            self.assertEqual(n, 2)
            self.assertIn("c01s01", run.bible()["characters"]["Mara"]["scenes"])


class TestLoop(unittest.TestCase):
    def test_full_loop_writes_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(tmp)
            client = MockOllama()
            n = run_loop(run, client=client)
            self.assertEqual(n, 3)
            self.assertEqual(len(run.scene_files()), 3)
            self.assertIn("Mara", run.bible()["characters"])
            self.assertIn("Compressed chapter: Mara progressed.", run.summary())
            self.assertIsNone(run.next_beat())
            steps = [e["step"] for e in run.ledger()]
            self.assertIn("draft_complete", steps)
            # a second invocation is a no-op, not a rewrite
            self.assertEqual(run_loop(run, client=client), 0)

    def test_glue_failure_flags_but_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(tmp)
            client = MockOllama()
            client.fail_models.add("qwen3.5:4b")  # glue dies; prose model lives
            data = write_scene(run, client, run.beats()[0])
            self.assertTrue(data["flag"])
            self.assertTrue(run.scene_path(run.beats()[0]).exists())

    def test_prose_failure_skips_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(tmp)
            client = MockOllama()
            client.fail_models.add("brie7b:latest")
            client.fail_models.add("qwen3.5:4b")
            n = run_loop(run, client=client, max_scenes=1)
            self.assertEqual(n, 1)  # flagged placeholder, run continued
            self.assertEqual(len(run.flags()) > 0, True)
            self.assertEqual(run.next_beat().id, "c01s02")


if __name__ == "__main__":
    unittest.main()
