#!/usr/bin/env python3
"""Turn an NMMO3 strategy context into dynamic reward weights and gamma."""

import json
import sys
import fcntl
import http.client
import os
import urllib.request
import urllib.error

from strategy_prompt import build_strategy_prompt
from strategy_response import parse_strategy_response

OLLAMA_URL = os.getenv("NMMO3_QWEN3_URL", "http://127.0.0.1:11434/api/generate")
MODEL = os.getenv("NMMO3_QWEN3_MODEL", "qwen3:4b-thinking")
REQUEST_TIMEOUT = float(os.getenv("NMMO3_QWEN3_TIMEOUT", "180"))
CONNECTION_RETRIES = 3
def query_ollama(strategy_context):
    prompt = build_strategy_prompt(strategy_context)
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps({
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "temperature": 0.2,
            "keep_alive": "10m",
            "format": "json",
            "options": {
                "num_ctx": 2048,
                "num_predict": 256,
                "temperature": 0.2,
            },
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with open("/tmp/nmmo3-qwen3.lock", "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        for attempt in range(CONNECTION_RETRIES):
            try:
                with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                    payload = json.load(response)
                    input_tokens = payload["prompt_eval_count"]
                    output_tokens = payload["eval_count"]
                    print(f"Input tokens: {input_tokens}", file=sys.stderr)
                    print(f"Output tokens: {output_tokens}", file=sys.stderr)
                    print(f"Total tokens: {input_tokens + output_tokens}", file=sys.stderr)
                break
            except urllib.error.URLError as error:
                refused = isinstance(error.reason, ConnectionRefusedError)
                if not refused or attempt == CONNECTION_RETRIES - 1:
                    raise
                import time
                time.sleep(2 ** attempt)
        else:
            raise urllib.error.URLError("Ollama connection retry limit reached")
    strategy = parse_strategy_response(payload.get("response", ""))
    strategy["llm_input_tokens"] = int(input_tokens)
    strategy["llm_output_tokens"] = int(output_tokens)
    return strategy


def main():
    context = sys.stdin.read()
    if not context.strip():
        raise SystemExit("empty NMMO3 strategy context")
    try:
        result = query_ollama(context)
    except (
        urllib.error.URLError,
        http.client.RemoteDisconnected,
        TimeoutError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise SystemExit(1)
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
