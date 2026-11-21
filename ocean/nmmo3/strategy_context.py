"""Validated, compact NMMO3 strategic context (no private keys or raw observations)."""
from dataclasses import dataclass, field, asdict
from typing import Mapping
import json
import math

OBJECTIVES = ("survive", "train_combat", "harvest", "equip", "trade", "explore")
CHANNELS = ("death", "pioneer", "combat_level", "profession_level",
            "attack_equipment", "defense_equipment", "tool_equipment",
            "market_purchase", "market_sale")

def finite(value, default=0.0):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError, OverflowError):
        return default

@dataclass
class StrategyContext:
    tick: int
    episode: int = 0
    alive_agents: int = 0
    num_agents: int = 0
    mean_health_fraction: float = 0.0
    nearby_enemy_count: int = 0
    nearby_resource_count: int = 0
    recent_events: dict = field(default_factory=dict)
    recent_base_return: float = 0.0
    current_objective: str = "survive"

    def __post_init__(self):
        self.tick = max(0, int(self.tick))
        self.episode = max(0, int(self.episode))
        self.num_agents = max(0, int(self.num_agents))
        self.alive_agents = max(0, min(int(self.alive_agents), self.num_agents))
        self.nearby_enemy_count = max(0, int(self.nearby_enemy_count))
        self.nearby_resource_count = max(0, int(self.nearby_resource_count))
        self.mean_health_fraction = max(0.0, min(1.0, finite(self.mean_health_fraction)))
        self.recent_base_return = finite(self.recent_base_return)
        self.current_objective = self.current_objective if self.current_objective in OBJECTIVES else "survive"
        events = self.recent_events if isinstance(self.recent_events, Mapping) else {}
        self.recent_events = {k: finite(events.get(k, 0.0)) for k in CHANNELS}

    def to_json(self):
        return json.dumps(asdict(self), separators=(",", ":"), allow_nan=False)
