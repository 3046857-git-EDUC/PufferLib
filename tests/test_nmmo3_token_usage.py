import contextlib
import io
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ocean", "nmmo3"))

import gemini_strategy
import gpt_strategy
import ollama_strategy
from strategy_cli import run_strategy_cli
from strategy_context import CHANNELS, StrategyContext


def guidance_json():
    return json.dumps({
        "objective": "survive",
        "reward_preferences": {key: 0.0 for key in CHANNELS},
        "horizon": 20,
    })


class TokenUsageTests(unittest.TestCase):
    def test_cli_emits_c_runtime_schema(self):
        context = StrategyContext(tick=10)
        stdout = io.StringIO()
        with (
            mock.patch.object(sys, "stdin", io.StringIO(context.to_json())),
            contextlib.redirect_stdout(stdout),
        ):
            run_strategy_cli(lambda _: (guidance_json(), 12, 7))

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["llm_input_tokens"], 12)
        self.assertEqual(payload["llm_output_tokens"], 7)
        self.assertEqual(payload["horizon"], 20)
        self.assertEqual(set(payload["reward_preferences"]), set(CHANNELS))

    def test_gpt_reports_provider_usage(self):
        body = json.dumps({
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
            "choices": [{"message": {"content": guidance_json()}}],
        }).encode()
        stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            mock.patch.object(gpt_strategy.urllib.request, "urlopen", return_value=io.BytesIO(body)),
            contextlib.redirect_stderr(stderr),
        ):
            result = gpt_strategy._request_gpt_response(StrategyContext(tick=1))

        self.assertEqual(result[1:], (12, 7))
        self.assertEqual(stderr.getvalue().splitlines(), [
            "Input tokens: 12", "Output tokens: 7", "Total tokens: 19",
        ])

    def test_ollama_reports_provider_usage(self):
        body = json.dumps({
            "prompt_eval_count": 12,
            "eval_count": 7,
            "message": {"content": guidance_json()},
        }).encode()
        stderr = io.StringIO()
        with (
            mock.patch.object(
                ollama_strategy.urllib.request, "urlopen", return_value=io.BytesIO(body)),
            contextlib.redirect_stderr(stderr),
        ):
            result = ollama_strategy._request_ollama_response(StrategyContext(tick=1))

        self.assertEqual(result[1:], (12, 7))
        self.assertEqual(stderr.getvalue().splitlines(), [
            "Input tokens: 12", "Output tokens: 7", "Total tokens: 19",
        ])

    def test_gemini_total_usage_maps_to_input_and_output(self):
        usage = SimpleNamespace(prompt_token_count=12, total_token_count=19)
        self.assertEqual(gemini_strategy._read_token_counts(usage), (12, 7))


if __name__ == "__main__":
    unittest.main()
