import json
import math


REWARD_WEIGHT_KEYS = (
    "reward_combat_level",
    "reward_prof_level",
    "reward_item_level",
    "reward_market",
    "reward_death",
)
EXPECTED_RESPONSE_KEYS = {"reward_weights", "gamma_exploration", "horizon_ticks"}
MAX_HORIZON_TICKS = 50
MIN_GAMMA_EXPLORATION = 0.74
MAX_GAMMA_EXPLORATION = 0.9999999


def parse_strategy_response(response_text):
    response_text = response_text.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[-1]
        response_text = response_text.rsplit("```", 1)[0].strip()

    strategy = json.loads(response_text)
    if not isinstance(strategy, dict):
        raise ValueError("strategy response must be a JSON object")
    if set(strategy) != EXPECTED_RESPONSE_KEYS:
        raise ValueError(
            "strategy response must contain only reward_weights, gamma_exploration, and horizon_ticks"
        )

    reward_weights = strategy.get("reward_weights")
    if not isinstance(reward_weights, dict) or set(reward_weights) != set(REWARD_WEIGHT_KEYS):
        raise ValueError("reward_weights must contain exactly the supported weight keys")
    for key in REWARD_WEIGHT_KEYS:
        value = reward_weights.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"reward_weights.{key} must be a number")
        try:
            finite_value = math.isfinite(value)
        except OverflowError:
            finite_value = False
        if not finite_value:
            raise ValueError(f"reward_weights.{key} must be finite")
        if key == "reward_death":
            valid_range = -1.0 <= value <= 0.0
        else:
            valid_range = 0.0 <= value <= 1.0
        if not valid_range:
            raise ValueError(f"reward_weights.{key} is outside its allowed range")

    gamma_exploration = strategy.get("gamma_exploration")
    if isinstance(gamma_exploration, bool) or not isinstance(gamma_exploration, (int, float)):
        raise ValueError("gamma_exploration must be a number")
    try:
        finite_gamma = math.isfinite(gamma_exploration)
    except OverflowError:
        finite_gamma = False
    if not finite_gamma:
        raise ValueError("gamma_exploration must be finite")
    if not MIN_GAMMA_EXPLORATION < gamma_exploration < MAX_GAMMA_EXPLORATION:
        raise ValueError("gamma_exploration is outside its allowed range")

    horizon_ticks = strategy.get("horizon_ticks")
    if (isinstance(horizon_ticks, bool) or not isinstance(horizon_ticks, int)
            or horizon_ticks <= 0 or horizon_ticks > MAX_HORIZON_TICKS):
        raise ValueError("horizon_ticks must be an integer from 1 through 50")

    return strategy
