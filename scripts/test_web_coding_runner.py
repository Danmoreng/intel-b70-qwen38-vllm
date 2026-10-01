"""Measurement regression checks using small controlled counter/stream examples."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "web_runner", Path(__file__).with_name("run-web-coding-benchmark.py")
)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


class MeasurementTests(unittest.TestCase):
    def test_output_budget_retains_reasoning_and_respects_common_window(self):
        payload = {
            "model": "test",
            "messages": [
                {
                    "role": "assistant",
                    "content": "code",
                    "reasoning_content": "retained",
                }
            ],
            "tools": [],
            "chat_template_kwargs": {"preserve_thinking": True},
            "max_tokens": 16384,
        }
        with patch.object(
            R, "http", return_value={"count": 190084, "max_model_len": 262144}
        ) as http:
            result = R.bound_output("test", payload)
        self.assertEqual(payload["max_tokens"], 10620)
        self.assertEqual(result["context_limit"], 200704)
        self.assertEqual(payload["messages"][0]["reasoning_content"], "retained")
        rendered = http.call_args.args[2]["messages"][0]
        self.assertEqual(rendered["reasoning"], "retained")
        self.assertNotIn("reasoning_content", rendered)
        with patch.object(
            R, "http", return_value={"count": 200704, "max_model_len": 262144}
        ):
            with self.assertRaisesRegex(RuntimeError, "window exhausted"):
                R.bound_output("test", payload)

    def test_missing_visible_stream_token_does_not_invent_client_rate(self):
        rows = [
            {
                "usage": {"prompt_tokens": 100, "completion_tokens": 1},
                "native": {
                    "prompt_tokens": 100,
                    "prefill_tokens": 100,
                    "prefill_seconds": 1,
                    "decode_seconds": 0,
                },
                "wall_s": 1,
                "ttft_s": None,
            }
        ]
        x = R.summarize(rows)
        self.assertIsNone(x["overall"]["client_decode_tps"])
        self.assertIsNone(x["overall"]["decode_tps"])

    def test_edit_rejects_ambiguous_spans_and_protected_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            p = root / "src/game.js"
            p.write_text("same same")
            result = R.execute(
                root,
                "edit_file",
                {"path": "src/game.js", "old_text": "same", "new_text": "other"},
                1,
                root,
                1,
            )
            self.assertIn("error", result)
            self.assertEqual(p.read_text(), "same same")
            p.write_text("first second")
            result = R.execute(
                root,
                "edit_file",
                {"path": "src/game.js", "old_text": "first", "new_text": "other"},
                1,
                root,
                1,
            )
            self.assertNotIn("error", result)
            self.assertEqual(p.read_text(), "other second")
            result = R.execute(
                root,
                "write_file",
                {"path": "specs/config.md", "content": "changed"},
                1,
                root,
                1,
            )
            self.assertIn("protected", result["error"])
            self.assertFalse((root / "specs/config.md").exists())

    def test_tool_history_removes_incidental_timings_and_host_paths(self):
        value = {
            "output": "✔ physics (1.239ms)\nℹ duration_ms 50.3\n"
            + str(R.REPO / "src/game.js")
            + " expected 0.35"
        }
        out = R.normalize_tool_result(value, Path("/tmp/isolated-project"))
        self.assertEqual(
            out["output"],
            "✔ physics (<time>ms)\nℹ duration_ms <time>\n/benchmark/src/game.js expected 0.35",
        )
        self.assertIn("1.239", value["output"])

    def test_weighted_band_rates_exclude_cached_prefill_and_first_output(self):
        rows = [
            {
                "usage": {"prompt_tokens": 15000, "completion_tokens": 101},
                "native": {
                    "prompt_tokens": 15000,
                    "cached_tokens": 14000,
                    "prefill_tokens": 1000,
                    "prefill_seconds": 2,
                    "decode_seconds": 2,
                },
                "wall_s": 5,
                "ttft_s": 3,
            },
            {
                "usage": {"prompt_tokens": 17000, "completion_tokens": 301},
                "native": {
                    "prompt_tokens": 17000,
                    "cached_tokens": 15000,
                    "prefill_tokens": 2000,
                    "prefill_seconds": 1,
                    "decode_seconds": 3,
                },
                "wall_s": 5,
                "ttft_s": 2,
            },
        ]
        x = R.summarize(rows)
        self.assertEqual(x["overall"]["prefill_tps"], 1000)
        self.assertEqual(x["overall"]["decode_tps"], 80)
        self.assertEqual(x["overall"]["post_first_tokens"], 400)
        self.assertEqual(x["bands"][0]["requests"], 0)
        self.assertIsNone(x["bands"][0]["decode_tps"])
        self.assertEqual(x["bands"][1]["requests"], 2)

    def test_accounting_waits_until_all_request_counters_agree(self):
        before = {"completed": 2, "generation_tokens": 200, "prompt_tokens": 2000}
        answers = [
            before,
            {"completed": 3, "generation_tokens": 200, "prompt_tokens": 3000},
            {"completed": 3, "generation_tokens": 250, "prompt_tokens": 3000},
        ]
        with (
            patch.object(R, "snapshot", side_effect=answers),
            patch.object(R.time, "sleep"),
        ):
            _, d = R.wait_accounted(
                "unused", before, {"completion_tokens": 50, "prompt_tokens": 1000}
            )
        self.assertEqual(
            d, {"completed": 1, "generation_tokens": 50, "prompt_tokens": 1000}
        )

    def test_concurrent_model_request_cannot_contaminate_band(self):
        with patch.object(
            R,
            "snapshot",
            return_value={
                "completed": 2,
                "generation_tokens": 70,
                "prompt_tokens": 2000,
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "concurrent traffic"):
                R.wait_accounted(
                    "unused",
                    {"completed": 0, "generation_tokens": 0, "prompt_tokens": 0},
                    {"completion_tokens": 50, "prompt_tokens": 1000},
                )

    def test_stream_assembles_interleaved_tools_and_reasoning_without_losing_usage(
        self,
    ):
        events = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"reasoning_content": "Think."}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "a",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":',
                                    },
                                },
                                {
                                    "index": 1,
                                    "id": "b",
                                    "function": {"name": "run_tests", "arguments": "{"},
                                },
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 1, "function": {"arguments": "}"}},
                                {"index": 0, "function": {"arguments": '"src/a.js"}'}},
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20}},
        ]

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def __iter__(self):
                return iter(
                    [b"data: " + json.dumps(e).encode() + b"\n" for e in events]
                    + [b"data: [DONE]\n"]
                )

        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(R.urllib.request, "urlopen", return_value=Response()),
        ):
            a = R.stream("http://localhost", {}, Path(d) / "stream.jsonl")
        self.assertEqual(a["reasoning"], "Think.")
        self.assertEqual(a["message"]["reasoning_content"], "Think.")
        self.assertEqual(a["usage"]["completion_tokens"], 20)
        self.assertEqual(
            a["message"]["tool_calls"][0]["function"]["arguments"],
            '{"path":"src/a.js"}',
        )
        self.assertEqual(a["message"]["tool_calls"][1]["function"]["arguments"], "{}")
        self.assertGreaterEqual(a["ttft_s"], 0)


if __name__ == "__main__":
    unittest.main()
