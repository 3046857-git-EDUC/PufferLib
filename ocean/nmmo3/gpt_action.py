#!/usr/bin/env python3
import json
import os
import sys
import urllib.error
import urllib.request

from action_prompt import build_action_prompt

MODEL = os.getenv("NMMO3_GPT_MODEL", "gpt-4o-mini")
TIMEOUT = float(os.getenv("NMMO3_GPT_TIMEOUT", "30"))


def main():
    context = sys.stdin.read()
    if not context.strip():
        raise SystemExit("empty NMMO3 action context")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": build_action_prompt(context)}],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is missing")
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            result = json.load(response)
        action = json.loads(result["choices"][0]["message"]["content"])
        code = int(action["action_code"])
        if code < 0 or code > 25 or code in (6, 7):
            raise ValueError("invalid action_code")
        print(json.dumps({"action_code": code, "confidence": float(action.get("confidence", 0.0))}))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, urllib.error.URLError) as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()