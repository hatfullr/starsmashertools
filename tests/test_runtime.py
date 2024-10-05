import unittest
import numpy as np
import basetest
import starsmashertools
import os

class TestRuntime(basetest.BaseTest):
    def setUp(self):
        curdir = os.path.dirname(__file__)
        self.simulation = starsmashertools.get_simulation(
            os.path.join(curdir, 'data'),
        )
    
    def test_get_wall_time(self):
        self.assertAlmostEqual(
            self.simulation.runtime.get_wall_time().value,
            starsmashertools.lib.units.Unit(6.891914845, 's'),
            places = 6,
        )


if __name__ == "__main__":
    unittest.main(failfast = True)
