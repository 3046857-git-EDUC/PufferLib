#!/usr/bin/env python3
import json
import os
import sys
import urllib.error
import urllib.request

from action_prompt import build_action_prompt

MODEL = os.getenv("NMMO3_GEMINI_MODEL", "gemini-3-flash-preview")
TIMEOUT = float(os.getenv("NMMO3_GEMINI_TIMEOUT", "60"))


def main():
    context = sys.stdin.read()
    if not context.strip():
        raise SystemExit("empty NMMO3 action context")
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is missing")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={key}"
    payload = {
        "contents": [{"parts": [{"text": build_action_prompt(context)}]}],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.2},
    }
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            result = json.load(response)
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        action = json.loads(text)
        code = int(action["action_code"])
        if code < 0 or code > 25 or code in (6, 7):
            raise ValueError("invalid action_code")
        print(json.dumps({"action_code": code, "confidence": float(action.get("confidence", 0.0))}))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, urllib.error.URLError) as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()