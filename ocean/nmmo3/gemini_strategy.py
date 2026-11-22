"""Gemini high-level reward guidance for HAL (never chooses primitive actions).

Install: pip install google-genai
Set GEMINI_API_KEY (or GOOGLE_API_KEY); optionally GEMINI_MODEL.
"""
import os
import sys
from strategy_context import StrategyContext
from strategy_prompt import build_messages
from reward_shaping import parse_guidance, safe_guidance
from strategy_cli import print_token_usage, run_strategy_cli


def _read_token_counts(usage):
    if usage is None:
        raise ValueError("Gemini response is missing token usage metadata")
    input_tokens = getattr(usage, "prompt_token_count", None)
    if type(input_tokens) is not int or input_tokens < 0:
        raise ValueError("Gemini response is missing a valid prompt token count")
    total_tokens = getattr(usage, "total_token_count", None)
    if total_tokens is not None:
        if type(total_tokens) is not int or total_tokens < input_tokens:
            raise ValueError("Gemini response contains an invalid total token count")
        output_tokens = total_tokens - input_tokens
    else:
        output_tokens = getattr(usage, "response_token_count", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "candidates_token_count", None)
        if type(output_tokens) is not int or output_tokens < 0:
            raise ValueError("Gemini response is missing a valid output token count")
        for field in ("thoughts_token_count", "tool_use_prompt_token_count"):
            extra_tokens = getattr(usage, field, 0) or 0
            if type(extra_tokens) is not int or extra_tokens < 0:
                raise ValueError(f"Gemini response contains an invalid {field}")
            output_tokens += extra_tokens
    return input_tokens, output_tokens


def _request_gemini_response(context: StrategyContext, model=None, timeout=30):
    key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not key:
        raise RuntimeError('Set GEMINI_API_KEY or GOOGLE_API_KEY')
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError('Install the Gemini SDK: pip install google-genai') from exc

    messages = build_messages(context)
    model = model or os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    with genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout * 1000))) as client:
        result = client.models.generate_content(
            model=model,
            contents=messages[1]['content'],
            config=types.GenerateContentConfig(
                system_instruction=messages[0]['content'],
                temperature=0,
                response_mime_type='application/json',
            ),
        )
    input_tokens, output_tokens = _read_token_counts(result.usage_metadata)
    print_token_usage(input_tokens, output_tokens)
    return result.text or "", input_tokens, output_tokens


def request_gemini_guidance(context: StrategyContext, model=None, timeout=30):
    """Return validated Guidance; raise on provider or validation failures."""
    response_text, _, _ = _request_gemini_response(context, model, timeout)
    return parse_guidance(response_text)


def request_or_fallback(context: StrategyContext, model=None, timeout=30):
    """Fail closed to neutral/survival preferences; return (guidance, error)."""
    try:
        return request_gemini_guidance(context, model, timeout), None
    except Exception as exc:
        return safe_guidance(), f'{type(exc).__name__}: {exc}'


def main():
    try:
        run_strategy_cli(_request_gemini_response)
    except (OSError, RuntimeError, ValueError, ImportError, KeyError,
            TimeoutError) as exc:
        print(f'Gemini strategy failed: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
