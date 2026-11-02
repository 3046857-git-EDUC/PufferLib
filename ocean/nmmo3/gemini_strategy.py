#!/usr/bin/env python3
#"""Turn an NMMO3 strategy context into one validated action decision using Google Gemini."""

import json
import os
import sys

try:
    from google import genai
    from google.genai import types
except ImportError as error:
    raise ImportError(
        f"Gemini strategy requires google-genai in {sys.executable}; "
        f"install it with '{sys.executable} -m pip install google-genai'"
    ) from error

from strategy_prompt import build_strategy_prompt

MODEL = os.getenv("NMMO3_GEMINI_MODEL", "gemini-3.5-flash-lite")
REQUEST_TIMEOUT = float(os.getenv("NMMO3_GEMINI_TIMEOUT", "60"))


def get_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if key and key.strip() and not key.startswith("your_") and not key.startswith("AIzaSyYourActual"):
        return key.strip()
    # Check .env file in workspace root or current directory
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"),
        os.path.join(os.getcwd(), ".env"),
        os.path.expanduser("~/.env"),
    ]
    for env_file in candidates:
        if os.path.isfile(env_file):
            try:
                with open(env_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY="):
                            val = line.split("=", 1)[1].strip().strip("'\"")
                            if val and not val.startswith("your_") and not val.startswith("AIzaSyYourActual"):
                                return val
            except Exception:
                pass
    return None


def parse_strategy_response(response_text):
    response_text = response_text.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[-1]
        response_text = response_text.rsplit("```", 1)[0].strip()
    permitted_strategies = {
        "explore", "harvest", "equip", "trade", "engage_NPC",
        "avoid_combat", "retreat", "recover",
    }
    strategy = json.loads(response_text)
    if not isinstance(strategy, dict):
        raise ValueError("strategy response must be a JSON object")
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


def query_gemini(strategy_context):
    api_key = get_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing or invalid.")

    prompt = build_strategy_prompt(strategy_context)
    client = genai.Client(api_key=api_key, vertexai=True)
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
                http_options=types.HttpOptions(timeout=int(REQUEST_TIMEOUT * 1000)),
            ),
        )
        strategy = parse_strategy_response(response.text)
        usage = response.usage_metadata
        if usage is None:
            raise ValueError("Gemini response is missing token usage metadata")
        input_tokens = int(usage.prompt_token_count)
        output_tokens = int(usage.candidates_token_count)
        print(f"Input tokens: {input_tokens}", file=sys.stderr)
        print(f"Output tokens: {output_tokens}", file=sys.stderr)
        print(f"Total tokens: {input_tokens + output_tokens}", file=sys.stderr)
        strategy["llm_input_tokens"] = input_tokens
        strategy["llm_output_tokens"] = output_tokens
        return strategy
    finally:
        client.close()


def main():
    context = sys.stdin.read()
    if not context.strip():
        raise SystemExit("Empty NMMO3 strategy context")
    try:
        result = query_gemini(context)
        print(json.dumps(result))
    except Exception as error:
        print(f"Gemini request failed: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
