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
        with self.assertRaises(starsmashertools.lib.simulation.Simulation.InvalidDirectoryError):
            starsmashertools.lib.simulation.Simulation('this is not a real directory, at all')

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

        # Not allowed to join 2 of the same simulations together
        with self.assertRaises(ValueError):
            sim1.join(sim2)

        # This simulation has out002.sph and out003.sph, while sim1 has
        # out000.sph and out001.sph.
        sim2 = starsmashertools.lib.simulation.Simulation(
            os.path.join(os.path.dirname(self.simdir), 'data2'),
        )

        sim1.join(sim2)
        
        self.assertIn(sim2, sim1.joined_simulations)
        self.assertIn(sim1, sim2.joined_simulations)

        # This should do nothing
        sim1.join(sim2)

        self.assertIn(sim2, sim1.joined_simulations)
        self.assertIn(sim1, sim2.joined_simulations)

        self.assertEqual(1, len(sim1.joined_simulations))
        self.assertEqual(1, len(sim2.joined_simulations))

        self.assertEqual(sim1, sim2.joined_simulations[0])
        self.assertEqual(sim2, sim1.joined_simulations[0])
        
        sim1.split()

        self.assertNotIn(sim2, sim1.joined_simulations)
        self.assertNotIn(sim1, sim2.joined_simulations)

        # Everything should work in reverse, too
        sim2.join(sim1)

        self.assertIn(sim2, sim1.joined_simulations)
        self.assertIn(sim1, sim2.joined_simulations)

        self.assertEqual(1, len(sim1.joined_simulations))
        self.assertEqual(1, len(sim2.joined_simulations))

        self.assertEqual(sim1, sim2.joined_simulations[0])
        self.assertEqual(sim2, sim1.joined_simulations[0])


        sim2.split()

        self.assertNotIn(sim2, sim1.joined_simulations)
        self.assertNotIn(sim1, sim2.joined_simulations)



        # Splitting on opposite simulations should work
        sim1.join(sim2)
        self.assertIn(sim2, sim1.joined_simulations)
        self.assertIn(sim1, sim2.joined_simulations)

        sim2.split()

        self.assertNotIn(sim2, sim1.joined_simulations)
        self.assertNotIn(sim1, sim2.joined_simulations)


        # Test some methods
        sim1.join(sim2)

        self.assertEqual(sim1.get_initialfile(), sim2.get_initialfile())
        self.assertEqual(sim1.get_output(), sim2.get_output())
        #self.assertEqual(4, len(sim2.get_output()))

        out1 = sim1.get_outputfiles()
        out2 = sim2.get_outputfiles()
        for i, name in enumerate(['out000.sph', 'out001.sph', 'out002.sph', 'out003.sph']):
            self.assertEqual(os.path.basename(out1[i]), os.path.basename(out2[i]))
            self.assertEqual(name, os.path.basename(out1[i]))


        self.assertEqual(
            'out002.sph',
            os.path.basename(sim2.get_outputfiles(include_joined = False)[0]),
        )
        self.assertEqual(
            'out000.sph',
            os.path.basename(sim1.get_outputfiles(include_joined = False)[0]),
        )

        # I built the simulations to have identical log files
        self.assertEqual(
            sim1.get_logfiles()[0],
            sim2.get_logfiles()[0],
        )

        # Make sure to restore the original behavior
        sim1.split()
        

if __name__ == "__main__":
    unittest.main(failfast=True)
