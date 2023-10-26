import unittest
import starsmashertools.helpers.argumentenforcer
import os
import numpy as np

"""
Add tests here for functions tagged with the wrapper "@api".
"""

curdir = os.getcwd()

class TestSimulation(unittest.TestCase):
    def setUp(self):
        path = os.path.join(curdir, 'data')
        self.simulation = starsmashertools.get_simulation(path)

    def tearDown(self):
        del self.simulation

    def test_get_output(self):
        self.assertEqual(len(self.simulation.get_output()), 2)
        
    def test_get_ejected_mass(self):
        self.assertEqual(self.simulation.get_ejected_mass(indices=np.array([0])), 0)
        mej0, mej1 = self.simulation.get_ejected_mass()
        self.assertEqual(mej0, 0)
        self.assertEqual(mej1, 0)


if __name__ == "__main__":
    unittest.main(failfast=True)
