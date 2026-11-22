"""OpenAI GPT strategic reward preferences using Python standard library.

Environment: OPENAI_API_KEY, optional OPENAI_MODEL (default gpt-5.4-nano).
Requires network access at runtime; no calls occur during import.
"""
import json
import os
import sys
import urllib.error
import urllib.request
from strategy_context import StrategyContext
from strategy_prompt import build_messages
from reward_shaping import parse_guidance, safe_guidance
from strategy_cli import print_token_usage, run_strategy_cli


def _request_gpt_response(context: StrategyContext, model=None, timeout=30):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before requesting GPT guidance")
    model = model or os.environ.get("OPENAI_MODEL", "gpt-5.4-nano")
    payload = {"model": model, "messages": build_messages(context),
               "temperature": 0, "response_format": {"type": "json_object"}}
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key,
                 "Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
        usage = result.get("usage")
        if not isinstance(usage, dict):
            raise ValueError("OpenAI response is missing token usage metadata")
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        if (type(input_tokens) is not int or input_tokens < 0
                or type(output_tokens) is not int or output_tokens < 0):
            raise ValueError("OpenAI response contains invalid token counts")
        print_token_usage(input_tokens, output_tokens)
        content = ""
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                content = message.get("content", "") or ""
        return content, input_tokens, output_tokens


def request_gpt_guidance(context: StrategyContext, model=None, timeout=30):
    content, _, _ = _request_gpt_response(context, model, timeout)
    return parse_guidance(content)


def request_or_fallback(context: StrategyContext, model=None):
    try:
        return request_gpt_guidance(context, model=model), None
    except (RuntimeError, ValueError, KeyError, IndexError,
            urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return safe_guidance(), str(exc)


def main():
    try:
        run_strategy_cli(_request_gpt_response)
    except (OSError, RuntimeError, ValueError, KeyError, IndexError,
            urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"GPT strategy failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
