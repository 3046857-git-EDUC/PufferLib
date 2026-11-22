"""CLI bridge between NMMO3's C runtime and Python strategy providers."""

import json
import sys

from reward_shaping import parse_guidance
from strategy_context import StrategyContext


def print_token_usage(input_tokens: int, output_tokens: int) -> None:
    print(f"Input tokens: {input_tokens}", file=sys.stderr, flush=True)
    print(f"Output tokens: {output_tokens}", file=sys.stderr, flush=True)
    print(f"Total tokens: {input_tokens + output_tokens}", file=sys.stderr, flush=True)


def run_strategy_cli(request):
    context_data = json.load(sys.stdin)
    if not isinstance(context_data, dict):
        raise ValueError("NMMO3 strategy context must be a JSON object")
    context = StrategyContext(**context_data)
    response_text, input_tokens, output_tokens = request(context)
    if (type(input_tokens) is not int or input_tokens < 0
            or type(output_tokens) is not int or output_tokens < 0):
        raise ValueError("Provider returned invalid token counts")

    token_usage = {
        "llm_input_tokens": input_tokens,
        "llm_output_tokens": output_tokens,
    }
    try:
        guidance = parse_guidance(response_text)
    except ValueError as exc:
        print(f"Strategy response validation failed: {exc}", file=sys.stderr, flush=True)
        print(json.dumps({**token_usage, "error": str(exc)}), flush=True)
        return

    print(json.dumps({
        "objective": guidance.objective,
        "reward_preferences": guidance.preferences,
        "horizon": guidance.horizon,
        **token_usage,
    }, separators=(",", ":"), allow_nan=False), flush=True)
