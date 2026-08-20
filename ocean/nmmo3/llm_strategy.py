#!/usr/bin/env python3
"""Turn an NMMO3 strategy context into one validated action decision."""

import json
import sys
import urllib.request
import urllib.error

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3"
ACTION_MAP = {
    "MOVE_DOWN": 0,
    "MOVE_UP": 1,
    "MOVE_RIGHT": 2,
    "MOVE_LEFT": 3,
    "NOOP": 4,
    "ATTACK": 5,
    "USE_ITEM_1": 8,
    "USE_ITEM_2": 9,
    "USE_ITEM_3": 10,
    "USE_ITEM_4": 11,
    "USE_ITEM_5": 12,
    "USE_ITEM_6": 13,
    "USE_ITEM_7": 14,
    "USE_ITEM_8": 15,
    "USE_ITEM_9": 16,
    "USE_ITEM_0": 17,
    "USE_ITEM_MINUS": 18,
    "USE_ITEM_EQUALS": 19,
    "BUY": 20,
    "SELL": 21,
    "MOVE_DOWN_SHIFT": 22,
    "MOVE_UP_SHIFT": 23,
    "MOVE_RIGHT_SHIFT": 24,
    "MOVE_LEFT_SHIFT": 25,
}


def parse_strategy_response(response_text):
    strategy = {"goal": "", "priority": "", "action": "NOOP", "action_code": 4}
    for raw_line in response_text.splitlines():
        label, separator, value = raw_line.partition(":")
        if not separator:
            continue
        value = value.strip()
        if label.strip().upper() == "GOAL":
            strategy["goal"] = value
        elif label.strip().upper() == "PRIORITY":
            strategy["priority"] = value
        elif label.strip().upper() == "ACTION":
            action = value.upper().replace(" ", "_")
            strategy["action"] = action
            strategy["action_code"] = ACTION_MAP.get(action, 4)
    return strategy


def query_qwen3(strategy_context):
    prompt = f"""You are the strategic advisor for a GROUP of NMMO3 reinforcement-learning agents.

The following information was generated directly from the current NMMO3 environment state.
Your recommendation is a single shared group directive and will be applied to all agents.
Use GROUP_STATE as the authoritative summary; SELF and nearby entities are supporting detail.
Do not invent information. Choose one short-term strategy that maximizes survival and useful progress.
ACTION must be exactly one of: MOVE_DOWN, MOVE_UP, MOVE_RIGHT, MOVE_LEFT,
NOOP, ATTACK, USE_ITEM_1, USE_ITEM_2, USE_ITEM_3, USE_ITEM_4, USE_ITEM_5,
USE_ITEM_6, USE_ITEM_7, USE_ITEM_8, USE_ITEM_9, USE_ITEM_0, USE_ITEM_MINUS,
USE_ITEM_EQUALS, BUY, SELL, MOVE_DOWN_SHIFT, MOVE_UP_SHIFT,
MOVE_RIGHT_SHIFT, MOVE_LEFT_SHIFT.
Return exactly three lines and no markdown:
GOAL: <goal>
PRIORITY: <priority>
ACTION: <action>

{strategy_context}"""
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps({
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "temperature": 0.2,
            "keep_alive": "10m",
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    return parse_strategy_response(payload.get("response", ""))


def main():
    context = sys.stdin.read()
    if not context.strip():
        raise SystemExit("empty NMMO3 strategy context")
    try:
        result = query_qwen3(context)
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
        print(f"Qwen3 request failed: {error}", file=sys.stderr)
        result = {"goal": "SURVIVE", "priority": "NOOP", "action": "NOOP", "action_code": 4}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
