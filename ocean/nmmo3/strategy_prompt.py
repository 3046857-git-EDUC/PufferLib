"""Prompt for bounded LLM reward preferences; not primitive actions."""
from strategy_context import CHANNELS, OBJECTIVES, StrategyContext

SYSTEM_PROMPT = """You are an NMMO3 high-level reward-guidance planner.
You do NOT choose actions. Return only a JSON object:
{"objective":"survive","reward_preferences":{"death":1.0,"pioneer":0.0,
"combat_level":0.0,"profession_level":0.0,"attack_equipment":0.0,
"defense_equipment":0.0,"tool_equipment":0.0,"market_purchase":0.0,
"market_sale":0.0},"horizon":720}
Objective must be one of: survive, train_combat, harvest, equip, trade, explore.
Provide all nine reward_preferences keys, each a finite number between 0 and 1.
The death preference measures importance of avoiding death, never rewarding death.
Preferences describe strategic priorities, not probabilities or direct actions.
Only use evidence in the context. Avoid recommending infeasible market actions.
No markdown, explanations, or extra keys."""

def build_messages(context: StrategyContext):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Current NMMO3 snapshot:\\n" + context.to_json()},
    ]
