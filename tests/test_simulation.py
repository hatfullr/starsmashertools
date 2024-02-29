import unittest
import os
import starsmashertools.lib.simulation
import starsmashertools.lib.output
import starsmashertools
import basetest

# Test the functionality of the Simulation class
# This test depends on 

class TestSimulation(basetest.BaseTest):
    def setUp(self):
        curdir = os.path.dirname(__file__)
        self.simdir = os.path.join(curdir, 'data')

    def test_valid_directory(self):
        self.assertFalse(starsmashertools.lib.simulation.Simulation.valid_directory("."))

    def test_init(self):
        starsmashertools.lib.simulation.Simulation(self.simdir)

    def test_eq(self):
        sim1 = starsmashertools.lib.simulation.Simulation(self.simdir)
        sim2 = starsmashertools.lib.simulation.Simulation(self.simdir)
        self.assertEqual(sim1, sim2)

    def test_contains(self):
        s = starsmashertools.lib.simulation.Simulation(self.simdir)
        output = starsmashertools.lib.output.Output(
            os.path.join(s.directory, 'out000.sph'),
            s,
        )
        self.assertIn(os.path.join(s.directory, 'out000.sph'), s)
        self.assertIn(output, s)

        iterator = s.get_output_iterator()
        self.assertIn(iterator, s)

    def test_join(self):
        sim1 = starsmashertools.lib.simulation.Simulation(self.simdir)
        sim2 = starsmashertools.lib.simulation.Simulation(self.simdir)

        joined = sim1.join(sim2)
        joined.split()
        

if __name__ == "__main__":
    unittest.main(failfast=True)
