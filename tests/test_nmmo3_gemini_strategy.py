import contextlib
import io
import os
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from google.genai import errors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ocean", "nmmo3"))
import gemini_strategy


class GeminiStrategyTests(unittest.TestCase):
    def test_reports_usage_before_parsing_response(self):
        response = SimpleNamespace(
            text="not json",
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                total_token_count=16,
                response_token_count=4,
                thoughts_token_count=2,
            ),
        )
        client = SimpleNamespace(
            models=SimpleNamespace(generate_content=lambda **kwargs: response),
            close=lambda: None,
        )
        client_args = {}

        def create_client(**kwargs):
            client_args.update(kwargs)
            return client

        stderr = io.StringIO()
        with (
            mock.patch.object(gemini_strategy, "get_api_key", return_value="test-key"),
            mock.patch.object(gemini_strategy.genai, "Client", side_effect=create_client),
            contextlib.redirect_stderr(stderr),
            self.assertRaisesRegex(ValueError, "Expecting value"),
        ):
            gemini_strategy.query_gemini("{}")

        self.assertEqual(client_args, {"api_key": "test-key", "vertexai": True})
        self.assertEqual(stderr.getvalue().splitlines(), [
            "Input tokens: 10",
            "Output tokens: 6",
            "Total tokens: 16",
        ])

    def test_reads_legacy_candidate_token_metadata(self):
        usage = SimpleNamespace(
            prompt_token_count=10,
            total_token_count=None,
            candidates_token_count=4,
            thoughts_token_count=2,
            tool_use_prompt_token_count=1,
        )

        self.assertEqual(gemini_strategy._read_token_counts(usage), (10, 7))

    def test_explains_blocked_vertex_api_key(self):
        error = errors.ClientError(
            403,
            {
                "error": {
                    "status": "PERMISSION_DENIED",
                    "message": "API key blocked",
                    "details": [{"reason": "API_KEY_SERVICE_BLOCKED"}],
                },
            },
        )
        client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=mock.Mock(side_effect=error),
            ),
            close=mock.Mock(),
        )

        with (
            mock.patch.object(gemini_strategy, "get_api_key", return_value="test-key"),
            mock.patch.object(gemini_strategy.genai, "Client", return_value=client),
            self.assertRaisesRegex(RuntimeError, "Vertex AI"),
        ):
            gemini_strategy.query_gemini("{}")

        client.close.assert_called_once_with()
