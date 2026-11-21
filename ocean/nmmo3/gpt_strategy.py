"""OpenAI GPT strategic reward preferences using Python standard library.

Environment: OPENAI_API_KEY, optional OPENAI_MODEL (default gpt-4.1-mini).
Requires network access at runtime; no calls occur during import.
"""
import json
import os
import urllib.error
import urllib.request
from strategy_context import StrategyContext
from strategy_prompt import build_messages
from reward_shaping import parse_guidance, safe_guidance

def request_gpt_guidance(context: StrategyContext, model=None, timeout=30):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before requesting GPT guidance")
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
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
    content = result["choices"][0]["message"]["content"]
    return parse_guidance(content)

def request_or_fallback(context: StrategyContext, model=None):
    try:
        return request_gpt_guidance(context, model=model), None
    except (RuntimeError, ValueError, KeyError, IndexError,
            urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return safe_guidance(), str(exc)
