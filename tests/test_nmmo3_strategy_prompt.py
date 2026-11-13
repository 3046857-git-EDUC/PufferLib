import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ocean", "nmmo3"))
from strategy_prompt import build_strategy_prompt


class StrategyPromptTests(unittest.TestCase):
    def test_includes_return_trend_guidance_and_context(self):
        context = "RETURN_TRENDS scope=GROUP current_ticks=720 previous_ticks=720"
        prompt = build_strategy_prompt(context)

        self.assertIn(context, prompt)
        self.assertIn("Compare current and previous interval returns", prompt)
        for field in (
            "return_comb_lvl",
            "return_prof_lvl",
            "return_item_atk_lvl",
            "return_item_def_lvl",
            "return_item_tool_lvl",
            "return_market_buy",
            "return_market_sell",
            "return_death",
        ):
            self.assertIn(field, prompt)
        self.assertIn("prefer a trend toward zero or fewer penalties", prompt)


if __name__ == "__main__":
    unittest.main()
