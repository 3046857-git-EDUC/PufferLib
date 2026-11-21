"""Safe, optional HAL reward shaping; never mutate baseline rewards in place.

Input event components must be float values from the C Reward struct BEFORE
unsigned-byte encoding, and must be aligned with the resulting environment step.
"""
from dataclasses import dataclass
from typing import Mapping
from strategy_context import CHANNELS, OBJECTIVES, finite
import json

DEFAULT_PREFERENCES = {key: 0.0 for key in CHANNELS}
DEFAULT_PREFERENCES["death"] = 1.0

@dataclass(frozen=True)
class Guidance:
    objective: str
    preferences: dict
    horizon: int = 720

def parse_guidance(text: str) -> Guidance:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("LLM response must be a JSON object")
    objective = data.get("objective")
    if objective not in OBJECTIVES:
        raise ValueError("Invalid objective")
    values = data.get("reward_preferences")
    if not isinstance(values, dict) or set(values) != set(CHANNELS):
        raise ValueError("Exactly nine reward preferences are required")
    preferences = {}
    for key in CHANNELS:
        raw = values[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("Preferences must be numbers")
        number = finite(raw, default=float("nan"))
        if not 0 <= number <= 1:
            raise ValueError("Preferences must be finite and within [0, 1]")
        preferences[key] = number
    horizon = data.get("horizon", 720)
    if type(horizon) is not int or not 1 <= horizon <= 720:
        raise ValueError("Horizon must be an integer in [1,720]")
    return Guidance(objective, preferences, horizon)

def safe_guidance() -> Guidance:
    return Guidance("survive", DEFAULT_PREFERENCES.copy(), 720)

def shape_step(base_reward: float, components: Mapping[str, float],
               guidance: Guidance, strength: float = 0.2,
               pioneer_bonus: float = 0.0,
               tool_bonus: float = 0.0,
               shaping_cap: float = 2.0):
    """Return (HAL reward, shaping delta, diagnostic channels).

    Base reward is passed through unchanged. For death, only amplify negative
    death penalties; never turn a negative death outcome into a positive bonus.
    The bonus coefficients are HAL-only. Pioneer must represent a *new* tile
    event and tool rewards should be net signed improvements.
    """
    if not 0 <= strength <= 1 or not 0 <= shaping_cap:
        raise ValueError("Invalid shaping strength/cap")
    if pioneer_bonus < 0 or tool_bonus < 0:
        raise ValueError("Bonus coefficients must be nonnegative")
    r = {key: finite(components.get(key, 0.0)) for key in CHANNELS}
    contribution = {}
    for key in CHANNELS:
        value = r[key]
        if key == "death":
            value = min(0.0, value)
        contribution[key] = strength * guidance.preferences[key] * value
    contribution["pioneer_base_addition"] = pioneer_bonus * max(0.0, r["pioneer"])
    contribution["tool_base_addition"] = tool_bonus * r["tool_equipment"]
    delta = sum(contribution.values())
    delta = max(-shaping_cap, min(shaping_cap, delta))
    return finite(base_reward) + delta, delta, contribution

def reward_observation10(components: Mapping[str, float], scales: Mapping[str, float]):
    """10 float32-ready features: 9 normalized channels plus reserved zero.

    Use a separate float buffer; converting existing uint8 bytes to float
    cannot recover the original fractional values.
    """
    out = []
    for key in CHANNELS:
        scale = finite(scales.get(key, 1.0), 1.0)
        if scale <= 0:
            raise ValueError("Each normalization scale must be positive")
        value = finite(components.get(key, 0.0)) / scale
        out.append(max(-1.0, min(1.0, value)))
    return tuple(out) + (0.0,)
