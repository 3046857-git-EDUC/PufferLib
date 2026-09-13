#!/usr/bin/env python3
"""Turn an NMMO3 strategy context into one validated decision using OpenRouter."""

import json
import os
import sys
import urllib.error
import urllib.request

from strategy_prompt import build_strategy_prompt

MODEL = os.getenv("NMMO3_OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
REQUEST_TIMEOUT = float(os.getenv("NMMO3_OPENROUTER_TIMEOUT", "60"))
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def provider_fallbacks_enabled():
    value = os.getenv("NMMO3_OPENROUTER_ALLOW_FALLBACKS", "1").lower()
    return value not in {"0", "false", "no"}


def get_api_key():
    key = os.getenv("OPENROUTER_API_KEY")
    if key and key.strip() and not key.startswith("your_"):
        return key.strip()
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"),
        os.path.join(os.getcwd(), ".env"),
        os.path.expanduser("~/.env"),
    ]
    for env_file in candidates:
        if not os.path.isfile(env_file):
            continue
        try:
            with open(env_file, "r") as file:
                for line in file:
                    line = line.strip()
                    if line.startswith("OPENROUTER_API_KEY="):
                        value = line.split("=", 1)[1].strip().strip("'\"")
                        if value and not value.startswith("your_"):
                            return value
        except OSError:
            pass
    return None


def parse_strategy_response(response_text):
    response_text = response_text.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[-1]
        response_text = response_text.rsplit("```", 1)[0].strip()
    strategy = json.loads(response_text)
    if not isinstance(strategy, dict):
        raise ValueError("strategy response must be a JSON object")
    permitted_strategies = {
        "explore", "harvest", "equip", "trade", "engage_NPC",
        "avoid_combat", "retreat", "recover",
    }
    if strategy.get("strategy_id") not in permitted_strategies:
        raise ValueError("invalid strategy_id")
    required_sections = (
        "strategy_parameters", "risk_parameters", "social_parameters",
        "contingency_parameters",
    )
    if any(section not in strategy or not isinstance(strategy[section], dict)
           for section in required_sections):
        raise ValueError("strategy response is missing a required section")
    if "confidence" not in strategy:
        raise ValueError("strategy response is missing confidence")
    return strategy


def query_openrouter(strategy_context):
    api_key = get_api_key()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable (or .env file) is missing.")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": build_strategy_prompt(strategy_context)}],
        "response_format": {"type": "json_object"},
        "provider": {"allow_fallbacks": provider_fallbacks_enabled()},
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/3046857-git-EDUC/PufferLib",
            "X-Title": "PufferLib NMMO3 Strategy",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            data = json.load(response)
        text = data["choices"][0]["message"]["content"]
        return parse_strategy_response(text)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        if error.code == 429:
            raise RuntimeError(
                "OpenRouter rate limit: Gemma free provider is temporarily unavailable. "
                "Retry later, set NMMO3_OPENROUTER_ALLOW_FALLBACKS=1, add your own "
                "provider key in OpenRouter, or choose another NMMO3_OPENROUTER_MODEL."
            ) from error
        raise RuntimeError(f"HTTP Error {error.code}: {error.reason} - {body}") from error


def main():
    context = sys.stdin.read()
    if not context.strip():
        raise SystemExit("Empty NMMO3 strategy context")
    try:
        print(json.dumps(query_openrouter(context)))
    except Exception as error:
        print(f"OpenRouter request failed: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)