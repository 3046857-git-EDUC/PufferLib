import json
import sys


json.load(sys.stdin)
print("Input tokens: 11", file=sys.stderr, flush=True)
print("Output tokens: 5", file=sys.stderr, flush=True)
print("Total tokens: 16", file=sys.stderr, flush=True)
print(json.dumps({
    "reward_preferences": {
        "death": 1.0,
        "pioneer": 0.0,
        "combat_level": 0.0,
        "profession_level": 0.0,
        "attack_equipment": 0.0,
        "defense_equipment": 0.0,
        "tool_equipment": 0.0,
        "market_purchase": 0.0,
        "market_sale": 0.0,
    },
    "horizon": 20,
    "llm_input_tokens": 11,
    "llm_output_tokens": 5,
}), flush=True)
