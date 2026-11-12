import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ocean", "nmmo3"))
from strategy_response import parse_strategy_response


class StrategyResponseTests(unittest.TestCase):
    def setUp(self):
        self.response = {
            "reward_weights": {
                "reward_combat_level": 1.0,
                "reward_prof_level": 0.8,
                "reward_item_level": 0.7,
                "reward_market": 0.2,
                "reward_death": -1.0,
            },
            "gamma_exploration": 0.85,
            "horizon_ticks": 50,
        }

    def test_accepts_reward_weights_and_horizon(self):
        parsed = parse_strategy_response(json.dumps(self.response))

        self.assertEqual(parsed, self.response)

    def test_rejects_missing_gamma(self):
        del self.response["gamma_exploration"]

        with self.assertRaisesRegex(ValueError, "gamma_exploration"):
            parse_strategy_response(json.dumps(self.response))

    def test_rejects_extra_response_keys(self):
        self.response["return_pioneer"] = 0

        with self.assertRaisesRegex(ValueError, "only reward_weights"):
            parse_strategy_response(json.dumps(self.response))

    def test_rejects_extra_weights(self):
        self.response["reward_weights"]["reward_exploration"] = 0.5

        with self.assertRaisesRegex(ValueError, "supported weight keys"):
            parse_strategy_response(json.dumps(self.response))

    def test_rejects_invalid_weight_ranges(self):
        self.response["reward_weights"]["reward_death"] = 0.1

        with self.assertRaisesRegex(ValueError, "outside its allowed range"):
            parse_strategy_response(json.dumps(self.response))

    def test_rejects_non_finite_weights(self):
        self.response["reward_weights"]["reward_combat_level"] = float("nan")

        with self.assertRaisesRegex(ValueError, "must be finite"):
            parse_strategy_response(json.dumps(self.response))

    def test_rejects_non_positive_horizon(self):
        self.response["horizon_ticks"] = 0

        with self.assertRaisesRegex(ValueError, "1 through 50"):
            parse_strategy_response(json.dumps(self.response))

    def test_rejects_horizon_over_50(self):
        self.response["horizon_ticks"] = 51

        with self.assertRaisesRegex(ValueError, "1 through 50"):
            parse_strategy_response(json.dumps(self.response))

    def test_rejects_gamma_outside_open_range(self):
        for gamma in (0.74, 0.9999999):
            with self.subTest(gamma=gamma):
                self.response["gamma_exploration"] = gamma
                with self.assertRaisesRegex(ValueError, "outside its allowed range"):
                    parse_strategy_response(json.dumps(self.response))


if __name__ == "__main__":
    unittest.main()
