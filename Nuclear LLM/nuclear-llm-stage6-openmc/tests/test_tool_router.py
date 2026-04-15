import unittest

from stage6_openmc.tool_router import route_query


class ToolRouterTest(unittest.TestCase):
    def test_openmc_route(self):
        route = route_query("Explain LOCA transient behavior", "LOCA")
        self.assertTrue(route["use_openmc"])
        self.assertEqual(route["route_type"], "OPENMC_SIMULATION")

    def test_no_simulation_route(self):
        route = route_query("Define neutron moderation", "neutron moderation")
        self.assertFalse(route["use_openmc"])
        self.assertEqual(route["route_type"], "NO_SIMULATION")


if __name__ == "__main__":
    unittest.main()
