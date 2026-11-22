"""Ollama local high-level reward guidance for HAL.

Set OLLAMA_HOST (default http://127.0.0.1:11434), OLLAMA_MODEL
(default qwen3:4b-thinking). No Python dependencies beyond standard library.
"""
import json
import os
import sys
import urllib.request
from strategy_context import StrategyContext
from strategy_prompt import build_messages
from reward_shaping import parse_guidance, safe_guidance
from strategy_cli import print_token_usage, run_strategy_cli


def _request_ollama_response(context: StrategyContext, model=None, timeout=60):
    """Call Ollama /api/chat and return content plus measured token usage."""
    base = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434').rstrip('/')
    if not base.startswith(('http://127.0.0.1:', 'http://localhost:', 'http://[::1]:')):
        raise ValueError('OLLAMA_HOST must be a local loopback endpoint')
    model = model or os.getenv('OLLAMA_MODEL', 'qwen3:4b-thinking')
    payload = {
        'model': model,
        'messages': build_messages(context),
        'stream': False,
        'format': 'json',
        'options': {'temperature': 0},
        'think': False,
    }
    req = urllib.request.Request(
        base + '/api/chat',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}, method='POST',
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    input_tokens = result.get('prompt_eval_count')
    output_tokens = result.get('eval_count')
    if (type(input_tokens) is not int or input_tokens < 0
            or type(output_tokens) is not int or output_tokens < 0):
        raise ValueError('Ollama response is missing valid token usage metadata')
    print_token_usage(input_tokens, output_tokens)
    content = result.get('message', {}).get('content', '')
    return content, input_tokens, output_tokens


def request_ollama_guidance(context: StrategyContext, model=None, timeout=60):
    """Call Ollama /api/chat, parse strict JSON and return Guidance."""
    content, _, _ = _request_ollama_response(context, model, timeout)
    return parse_guidance(content)


def request_or_fallback(context: StrategyContext, model=None, timeout=60):
    try:
        return request_ollama_guidance(context, model, timeout), None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return safe_guidance(), f'{type(exc).__name__}: {exc}'


def main():
    try:
        run_strategy_cli(_request_ollama_response)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f'Ollama strategy failed: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
