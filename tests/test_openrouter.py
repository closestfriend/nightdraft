"""OpenRouter client tests. Only the network is faked: a real HTTP server on
localhost plays OpenRouter, so request building, retry policy, message
translation and spend accounting all run for real."""

import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from book.clients import make_client  # noqa: E402
from book.loop import run_loop  # noqa: E402
from book.ollama import Ollama, OllamaError  # noqa: E402
from book.openrouter import BudgetExceeded, OpenRouter  # noqa: E402
from book.state import Run  # noqa: E402
from test_harness import OUTLINE, MockOllama  # noqa: E402

MSGS = [{"role": "user", "content": "hi"}]


class FakeOpenRouter:
    """Scripted stand-in for openrouter.ai. `replies` are (status, body) pairs
    served in order; every request is recorded as (path, auth header, json body)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, self.headers.get("Authorization"), body))
                status, payload = outer.replies.pop(0) if outer.replies else (
                    500, {"error": {"code": 500, "message": "script exhausted"}})
                if payload == "DROP":  # connection dies with no response at all
                    self.connection.close()
                    return
                data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/api/v1"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def completion(content="ok", cost=0.001, prompt=100, out=20, reasoning=0, tool_calls=None, finish=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return (200, {
        "id": "gen-1",
        "model": "m/x",
        "choices": [{"message": message, "finish_reason": finish or ("tool_calls" if tool_calls else "stop")}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": out,
            "total_tokens": prompt + out,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
            "cost": cost,
        },
    })


def tool_call(call_id, query):
    return {"id": call_id, "type": "function",
            "function": {"name": "search_manuscript", "arguments": json.dumps({"query": query})}}


def client_for(fake, tmp, **kw):
    return OpenRouter(api_key="test-key", base=fake.base, timeout=5, backoff_s=0,
                      usage_log=Path(tmp) / "usage.jsonl", **kw)


class TestRequestShape(unittest.TestCase):
    def test_plain_chat_sends_an_openai_shaped_request(self):
        with FakeOpenRouter([completion("Hello")]) as fake, tempfile.TemporaryDirectory() as tmp:
            msg = client_for(fake, tmp, max_tokens=4096).chat("m/x", MSGS, temperature=0.7, num_ctx=8192)
            path, auth, body = fake.requests[0]
            self.assertEqual(path, "/api/v1/chat/completions")
            self.assertEqual(auth, "Bearer test-key")
            # exact body: no Ollama-isms (options / num_ctx / think) leak through
            self.assertEqual(body, {"model": "m/x", "messages": MSGS, "temperature": 0.7, "max_tokens": 4096})
            self.assertEqual(msg, {"role": "assistant", "content": "Hello"})

    def test_schema_becomes_json_schema_response_format_and_pins_capable_providers(self):
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
        with FakeOpenRouter([completion('{"a": 1}')]) as fake, tempfile.TemporaryDirectory() as tmp:
            result = client_for(fake, tmp).chat_json("m/x", MSGS, schema=schema)
            body = fake.requests[0][2]
            self.assertEqual(body["response_format"]["type"], "json_schema")
            self.assertEqual(body["response_format"]["json_schema"]["schema"], schema)
            self.assertEqual(body["provider"], {"require_parameters": True})
            self.assertEqual(result, {"a": 1})

    def test_think_false_disables_reasoning_and_none_leaves_it_alone(self):
        # hidden reasoning is billed as output tokens: the think:false lesson, in dollars
        with FakeOpenRouter([completion(), completion()]) as fake, tempfile.TemporaryDirectory() as tmp:
            client = client_for(fake, tmp)
            client.chat("m/x", MSGS, think=False)
            client.chat("m/x", MSGS, think=None)
            self.assertEqual(fake.requests[0][2]["reasoning"], {"effort": "none"})
            self.assertNotIn("reasoning", fake.requests[1][2])

    def test_tool_results_are_paired_with_call_ids_in_order(self):
        # OpenAI-style APIs reject tool messages without tool_call_id; the harness's
        # tool loop sends bare {"role": "tool", "content": ...} messages.
        calls = [tool_call("call_a", "door"), tool_call("call_b", "Mara")]
        with FakeOpenRouter([completion("", tool_calls=calls), completion("done")]) as fake, \
                tempfile.TemporaryDirectory() as tmp:
            client = client_for(fake, tmp)
            msg = client.chat("m/x", MSGS, tools=[{"type": "function", "function": {"name": "search_manuscript"}}])
            self.assertEqual(msg["tool_calls"][0]["function"]["arguments"], '{"query": "door"}')  # string, as sent
            convo = MSGS + [msg, {"role": "tool", "content": "r1"}, {"role": "tool", "content": "r2"}]
            client.chat("m/x", convo)
            sent = fake.requests[1][2]["messages"]
            self.assertEqual([m["role"] for m in sent], ["user", "assistant", "tool", "tool"])
            self.assertEqual([c["id"] for c in sent[1]["tool_calls"]], ["call_a", "call_b"])
            self.assertEqual([m["tool_call_id"] for m in sent[2:]], ["call_a", "call_b"])


class TestAccounting(unittest.TestCase):
    def test_each_call_is_logged_with_tokens_and_cost(self):
        with FakeOpenRouter([completion(cost=0.0123, prompt=1500, out=400, reasoning=50)]) as fake, \
                tempfile.TemporaryDirectory() as tmp:
            client = client_for(fake, tmp)
            client.chat("m/x", MSGS)
            lines = [json.loads(l) for l in (Path(tmp) / "usage.jsonl").read_text().splitlines()]
            self.assertEqual(len(lines), 1)
            self.assertEqual(
                {k: lines[0][k] for k in ("model", "prompt_tokens", "completion_tokens", "reasoning_tokens", "cost")},
                {"model": "m/x", "prompt_tokens": 1500, "completion_tokens": 400,
                 "reasoning_tokens": 50, "cost": 0.0123},
            )
            self.assertAlmostEqual(client.spent_usd, 0.0123)

    def test_budget_stops_the_call_that_would_start_over_the_cap(self):
        with FakeOpenRouter([completion(cost=0.6), completion(cost=0.6), completion(cost=0.6)]) as fake, \
                tempfile.TemporaryDirectory() as tmp:
            client = client_for(fake, tmp, budget_usd=1.0)
            client.chat("m/x", MSGS)  # spent 0.6 < 1.0: allowed
            client.chat("m/x", MSGS)  # spent 1.2 now (overshoot is at most one call)
            with self.assertRaises(BudgetExceeded):
                client.chat("m/x", MSGS)
            self.assertEqual(len(fake.requests), 2)  # the refused call never hit the network

    def test_budget_survives_a_restart(self):
        # the cap is per run directory, not per process: tomorrow night starts from spent-so-far
        with FakeOpenRouter([completion()]) as fake, tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "usage.jsonl").write_text(
                json.dumps({"model": "m/x", "cost": 0.9}) + "\n" + json.dumps({"model": "m/x", "cost": 0.6}) + "\n"
            )
            fresh = client_for(fake, tmp, budget_usd=1.0)
            with self.assertRaises(BudgetExceeded):
                fresh.chat("m/x", MSGS)
            self.assertEqual(fake.requests, [])


class TestErrors(unittest.TestCase):
    def test_insufficient_credits_fails_fast_with_the_provider_message(self):
        reply = (402, {"error": {"code": 402, "message": "Insufficient credits"}})
        with FakeOpenRouter([reply, completion()]) as fake, tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(OllamaError, "Insufficient credits"):
                client_for(fake, tmp).chat("m/x", MSGS)
            self.assertEqual(len(fake.requests), 1)  # a 4xx is not retried

    def test_rate_limit_is_retried(self):
        limited = (429, {"error": {"code": 429, "message": "slow down"}})
        with FakeOpenRouter([limited, completion("fine")]) as fake, tempfile.TemporaryDirectory() as tmp:
            msg = client_for(fake, tmp).chat("m/x", MSGS)
            self.assertEqual(msg["content"], "fine")
            self.assertEqual(len(fake.requests), 2)

    def test_error_body_behind_http_200_is_an_error_not_a_blank_scene(self):
        boom = (200, {"error": {"code": 502, "message": "upstream boom"}})
        with FakeOpenRouter([boom, boom, boom]) as fake, tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(OllamaError, "upstream boom"):
                client_for(fake, tmp).chat("m/x", MSGS)
            self.assertEqual(len(fake.requests), 3)  # 5xx-class: retried, then given up


class TestUnattendedSafety(unittest.TestCase):
    """Failures an overnight run must convert into OllamaError (which halts the run
    cleanly) rather than raw exceptions (which become placeholder scenes)."""

    def test_garbage_bodies_are_retried_then_reported_as_ollama_errors(self):
        for junk in ("<html>502 bad gateway</html>", [1, 2], {"error": "just a string"}):
            with self.subTest(body=junk), FakeOpenRouter([(200, junk)] * 3) as fake, \
                    tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(OllamaError):
                    client_for(fake, tmp).chat("m/x", MSGS)
                self.assertEqual(len(fake.requests), 3)

    def test_upstream_reason_in_error_metadata_reaches_the_message(self):
        # "Provider returned error" alone cost a debugging round-trip; the real reason
        # ("temporarily rate-limited upstream") lives in error.metadata.raw
        err = {"code": 429, "message": "Provider returned error",
               "metadata": {"raw": "m/x is temporarily rate-limited upstream"}}
        for status in (200, 429):
            with self.subTest(status=status), FakeOpenRouter([(status, {"error": err})] * 3) as fake, \
                    tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(OllamaError, "rate-limited upstream"):
                    client_for(fake, tmp).chat("m/x", MSGS)

    def test_dropped_connection_is_retried(self):
        with FakeOpenRouter([(200, "DROP"), completion("recovered")]) as fake, tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(client_for(fake, tmp).chat("m/x", MSGS)["content"], "recovered")
            self.assertEqual(len(fake.requests), 2)

    def test_dropped_connections_end_as_ollama_error(self):
        with FakeOpenRouter([(200, "DROP")] * 3) as fake, tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OllamaError):
                client_for(fake, tmp).chat("m/x", MSGS)

    def test_output_cut_off_at_the_token_cap_is_an_error_but_still_billed(self):
        # A truncated outline or scene must never be accepted as complete.
        with FakeOpenRouter([completion('{"acts": [', cost=0.02, finish="length")]) as fake, \
                tempfile.TemporaryDirectory() as tmp:
            client = client_for(fake, tmp, max_tokens=4096)
            with self.assertRaisesRegex(OllamaError, "max_output_tokens"):
                client.chat_json("m/x", MSGS, schema={"type": "object"})
            self.assertAlmostEqual(client.spent_usd, 0.02)  # the tokens were paid for

    def test_empty_reply_at_the_cap_is_an_error(self):
        # e.g. a reasoning model that spent the whole cap thinking
        with FakeOpenRouter([completion("", finish="length")]) as fake, tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OllamaError):
                client_for(fake, tmp).chat("m/x", MSGS)


class TestBudgetHaltsEverywhere(unittest.TestCase):
    """BudgetExceeded must stop the run wherever it surfaces, not be flagged and forgotten."""

    def _run(self, tmp, out_of_money_when, **config):
        """out_of_money_when(model, messages, schema) -> True on the call that runs dry."""

        class Broke(MockOllama):
            def chat(self, model, messages, **kw):
                if out_of_money_when(model, messages, None):
                    raise BudgetExceeded("budget $1.00 reached")
                return super().chat(model, messages, **kw)

            def chat_json(self, model, messages, schema, **kw):
                if out_of_money_when(model, messages, schema):
                    raise BudgetExceeded("budget $1.00 reached")
                return super().chat_json(model, messages, schema, **kw)

        run = Run.create(Path(tmp), "b", "premise", config)
        run.write_outline(OUTLINE, reason="test")
        return run, run_loop(run, client=Broke())

    def test_running_out_during_the_consistency_check_halts_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            # only the consistency verdict runs dry: extraction would succeed if the loop carried on
            run, n = self._run(tmp, lambda model, msgs, schema: bool(schema) and "consistent" in schema.get("properties", {}))
            self.assertEqual(n, 0)
            self.assertEqual(run.scene_files(), [])
            steps = [e["step"] for e in run.ledger()]
            self.assertIn("run_halted", steps)
            self.assertNotIn("scene_accepted", steps)

    def test_running_out_at_chapter_end_halts_instead_of_carrying_on(self):
        # chapter 1 has two scenes; the chapter-compression call is the one that fails
        with tempfile.TemporaryDirectory() as tmp:
            run, n = self._run(tmp, lambda model, msgs, schema: "Compress" in msgs[0].get("content", ""))
            self.assertEqual(n, 2)  # both chapter-1 scenes were written and kept
            self.assertIn("run_halted", [e["step"] for e in run.ledger()])
            self.assertEqual(run.next_beat().id, "c02s01")  # chapter 2 never started

    def test_running_out_during_canon_extraction_halts_and_writes_nothing(self):
        def extraction(model, msgs, schema):
            return bool(schema) and "scene_summary" in schema.get("properties", {})

        with tempfile.TemporaryDirectory() as tmp:
            run, n = self._run(tmp, extraction)
            self.assertEqual(n, 0)
            self.assertEqual(run.scene_files(), [])
            self.assertIn("run_halted", [e["step"] for e in run.ledger()])

    def test_running_out_during_the_drift_check_halts_instead_of_carrying_on(self):
        def drift(model, msgs, schema):
            return bool(schema) and "on_course" in schema.get("properties", {})

        with tempfile.TemporaryDirectory() as tmp:
            run, n = self._run(tmp, drift, drift_every_chapters=1)
            self.assertEqual(n, 2)
            self.assertIn("run_halted", [e["step"] for e in run.ledger()])
            self.assertEqual(run.next_beat().id, "c02s01")


class TestWiring(unittest.TestCase):
    def test_default_provider_is_still_ollama(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Run.create(Path(tmp), "b", "premise")
            self.assertIsInstance(make_client(run), Ollama)
            self.assertNotIsInstance(make_client(run), OpenRouter)

    def test_missing_api_key_is_a_clear_error_not_a_401_at_2am(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {}, clear=True):
            run = Run.create(Path(tmp), "b", "premise", {"provider": "openrouter"})
            with self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"):
                make_client(run)

    def test_config_budget_and_usage_log_reach_the_client(self):
        with FakeOpenRouter([completion(cost=0.06), completion()]) as fake, tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}):
            run = Run.create(Path(tmp), "b", "premise",
                             {"provider": "openrouter", "openrouter_base": fake.base, "budget_usd": 0.05})
            client = make_client(run)
            client.chat("m/x", MSGS)
            self.assertTrue((run.root / "usage.jsonl").exists())
            with self.assertRaises(BudgetExceeded):
                client.chat("m/x", MSGS)

    def test_budget_exceeded_halts_the_run_instead_of_writing_placeholders(self):
        class OutOfMoney(MockOllama):
            def chat(self, model, messages, **kw):
                raise BudgetExceeded("budget $1.00 spent")

        with tempfile.TemporaryDirectory() as tmp:
            run = Run.create(Path(tmp), "b", "premise")
            run.write_outline(OUTLINE, reason="test")
            self.assertEqual(run_loop(run, client=OutOfMoney()), 0)
            self.assertEqual(run.scene_files(), [])
            steps = [e["step"] for e in run.ledger()]
            self.assertIn("run_halted", steps)
            failed = next(e for e in run.ledger() if e["step"] == "scene_failed")
            self.assertIn("budget", failed["error"])

    def test_prose_calls_honor_prose_think(self):
        class Spy(MockOllama):
            def __init__(self):
                super().__init__()
                self.think_by_model = {}

            def chat(self, model, messages, **kw):
                self.think_by_model[model] = kw.get("think", "unset")
                return super().chat(model, messages, **kw)

        with tempfile.TemporaryDirectory() as tmp:
            run = Run.create(Path(tmp), "b", "premise", {"prose_think": False, "glue_think": None})
            run.write_outline(OUTLINE, reason="test")
            spy = Spy()
            run_loop(run, client=spy, max_scenes=1)
            self.assertIs(spy.think_by_model[run.config["prose_model"]], False)


if __name__ == "__main__":
    unittest.main()
