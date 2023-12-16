import unittest
import starsmashertools.helpers.argumentenforcer
import os
import numpy as np
import basetest

"""
Add tests here for functions tagged with the wrapper "@api".
"""

curdir = os.getcwd()

class TestSimulation(basetest.BaseTest):
    def setUp(self):
        path = os.path.join(curdir, 'data')
        self.simulation = starsmashertools.get_simulation(path)

    def tearDown(self):
        del self.simulation

    def test_get_output(self):
        self.assertEqual(len(self.simulation.get_output()), 2)

if __name__ == "__main__":
    unittest.main(failfast=True)
