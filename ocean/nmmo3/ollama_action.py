#!/usr/bin/env python3
import json
import os
import sys
import urllib.error
import urllib.request

from action_prompt import build_action_prompt

OLLAMA_URL = os.getenv("NMMO3_OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
MODEL = os.getenv("NMMO3_OLLAMA_MODEL", "qwen3:4b-thinking")
TIMEOUT = float(os.getenv("NMMO3_OLLAMA_TIMEOUT", "180"))


def main():
    context = sys.stdin.read()
    if not context.strip():
        raise SystemExit("empty NMMO3 action context")
    payload = {
        "model": MODEL,
        "prompt": build_action_prompt(context),
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0.2, "num_ctx": 4096, "num_predict": 64},
    }
    request = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            result = json.load(response)
        action = json.loads(result.get("response", ""))
        if not isinstance(action, dict):
            raise ValueError("action response must be an object")
        code = int(action["action_code"])
        if code < 0 or code > 25 or code in (6, 7):
            raise ValueError("invalid action_code")
        print(json.dumps({"action_code": code, "confidence": float(action.get("confidence", 0.0))}))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, urllib.error.URLError) as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()