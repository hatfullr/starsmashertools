import unittest
import os
import starsmashertools.lib.simulation
import starsmashertools.lib.output
import starsmashertools
import warnings
import basetest

# Test the functionality of the Simulation class
# This test depends on 

class TestRelaxation(basetest.BaseTest):
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

    def test_get_binding_energy(self):
        import starsmashertools.lib.units
        output = self.simulation.get_output(0)
        Ebind = self.simulation.get_binding_energy(output)
        self.assertTrue(isinstance(Ebind, starsmashertools.lib.units.Unit))
        self.assertEqual(5.463207228488382e+46, Ebind.value)
        self.assertEqual(Ebind.label, 'cm*cm*g/s*s')

if __name__ == "__main__":
    unittest.main(failfast=True)
