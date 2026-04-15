import unittest

from stage6_openmc.intent_parser import parse_intent


class IntentParserTest(unittest.TestCase):
    def test_loca_intent(self):
        intent = parse_intent("What happens during LOCA?")
        self.assertEqual(intent.concept, "LOCA")
        self.assertEqual(intent.scenario_type, "accident")
        self.assertIn("k_eff", intent.requested_outputs)

    def test_conceptual_fallback(self):
        intent = parse_intent("Explain neutron moderation simply")
        self.assertEqual(intent.route_type, "NO_SIMULATION")
        self.assertEqual(intent.concept, "reactor physics")


if __name__ == "__main__":
    unittest.main()
