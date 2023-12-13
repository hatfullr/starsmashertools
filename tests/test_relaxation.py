import unittest
import os
import starsmashertools.lib.simulation
import starsmashertools.lib.output
import starsmashertools
import warnings

# Test the functionality of the Simulation class
# This test depends on 

class TestRelaxation(unittest.TestCase):
    def setUp(self):
        curdir = os.path.dirname(__file__)
        simdir = os.path.join(curdir, 'data')
        self.simulation = starsmashertools.get_simulation(simdir)

    def test_get_n(self):
        self.assertEqual(self.simulation.get_n(), 1038)

    def test_get_final_extents(self):
        extents = self.simulation.get_final_extents()
        self.assertAlmostEqual(
            float(extents.radius)/float(self.simulation.units.length),
            0.876398091009529,
        )

    def test_isPolytrope(self):
        self.assertTrue(self.simulation.isPolytrope)

    def test_get_children(self):
        warnings.filterwarnings(action='ignore')
        self.assertEqual(self.simulation.get_children(), [])
        warnings.resetwarnings()


if __name__ == "__main__":
    unittest.main(failfast=True)
