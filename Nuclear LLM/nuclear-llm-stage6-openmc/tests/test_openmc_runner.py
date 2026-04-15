import unittest

from stage6_openmc.intent_parser import parse_intent
from stage6_openmc.openmc_runner import run_openmc
from stage6_openmc.reactor_config_builder import build_config


class OpenMCRunnerTest(unittest.TestCase):
    def test_runner_is_deterministic(self):
        intent = parse_intent("What happens during LOCA?")
        config = build_config(intent)
        result_a = run_openmc(config)
        result_b = run_openmc(config)
        self.assertEqual(result_a, result_b)
        self.assertIn("k_eff", result_a)
        self.assertIn("flux", result_a)


if __name__ == "__main__":
    unittest.main()
