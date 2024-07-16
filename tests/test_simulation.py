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

    def test_get_output(self):
        s = starsmashertools.lib.simulation.Simulation(self.simdir)

        outputs = [
            starsmashertools.lib.output.Output(
                os.path.join(s.directory, filename),
                s,
            ) for filename in ['out000.sph', 'out001.sph', 'out002.sph']
        ]
        for i, output in enumerate(s.get_output()):
            self.assertEqual(outputs[i], output)

        for i, output in enumerate(outputs):
            self.assertEqual(output, s.get_output(i))
            
        self.assertEqual([outputs[0], outputs[-1]], s.get_output(indices = [0,-1]))
        self.assertEqual(outputs[None:2:None], s.get_output(stop = 2))
        self.assertEqual(outputs[None:-1:None], s.get_output(stop = -1))
        self.assertEqual(outputs[1:None:None], s.get_output(start = 1))
        self.assertEqual(outputs[None:None:2], s.get_output(step = 2))

        times = [output['t'] * output.simulation.units['t'] for output in outputs]
        
        self.assertEqual(outputs, s.get_output(time_range = [min(times), max(times)]))
        self.assertEqual(outputs[:2], s.get_output(time_range = [None, times[1]]))
        self.assertEqual(outputs[1:], s.get_output(time_range = [times[1], None]))
        self.assertEqual(outputs[1], s.get_output(time_range = [times[1], times[1]]))

        self.assertEqual(outputs, s.get_output(times = times))
        for i, time in enumerate(times):
            self.assertEqual(outputs[i], s.get_output(times = [time]))

    def test_join(self):
        sim1 = starsmashertools.lib.simulation.Simulation(self.simdir)
        
        # This simulation has out002.sph and out003.sph, while sim1 has
        # out000.sph and out001.sph.
        sim2 = starsmashertools.lib.simulation.Simulation(
            os.path.join(os.path.dirname(self.simdir), 'data2'),
        )

        # Not allowed to join 2 of the same simulations together
        with sim1.archive.nosave():
            with self.assertRaises(starsmashertools.lib.simulation.Simulation.JoinError):
                sim1.join(sim1)

        self.assertEqual(0, len(sim1.joined_simulations))
        self.assertEqual(0, len(sim2.joined_simulations))

        # Just to make sure...
        with sim2.archive.nosave():
            with self.assertRaises(starsmashertools.lib.simulation.Simulation.JoinError):
                sim2.join(sim2)

        self.assertEqual(0, len(sim1.joined_simulations))
        self.assertEqual(0, len(sim2.joined_simulations))

        with sim1.archive.nosave():
            with sim2.archive.nosave():
                self.assertEqual(1, sim1.archive._nosave_holders)
                self.assertEqual(1, sim2.archive._nosave_holders)
                
                sim1.join(sim2)
                self.assertEqual(1, len(sim1.joined_simulations))
                self.assertEqual(1, len(sim2.joined_simulations))
                self.assertIn(sim2, sim1.joined_simulations)
                self.assertIn(sim1, sim2.joined_simulations)
                self.assertEqual(sim1, sim2.joined_simulations[0])
                self.assertEqual(sim2, sim1.joined_simulations[0])

                # Once joined, joining should do nothing
                sim1.join(sim2)
                
                self.assertEqual(1, len(sim1.joined_simulations))
                self.assertEqual(1, len(sim2.joined_simulations))
                self.assertIn(sim2, sim1.joined_simulations)
                self.assertIn(sim1, sim2.joined_simulations)
                self.assertEqual(sim1, sim2.joined_simulations[0])
                self.assertEqual(sim2, sim1.joined_simulations[0])

                sim2.join(sim1)

                self.assertEqual(1, len(sim1.joined_simulations))
                self.assertEqual(1, len(sim2.joined_simulations))
                self.assertIn(sim2, sim1.joined_simulations)
                self.assertIn(sim1, sim2.joined_simulations)
                self.assertEqual(sim1, sim2.joined_simulations[0])
                self.assertEqual(sim2, sim1.joined_simulations[0])


        self.assertEqual(0, sim1.archive._nosave_holders)
        self.assertEqual(0, sim2.archive._nosave_holders)

        # Make sure the nosave() method worked
        self.assertEqual(0, len(sim1.joined_simulations))
        self.assertEqual(0, len(sim2.joined_simulations))

        # Try the same but in reverse
        with sim1.archive.nosave():
            with sim2.archive.nosave():
                sim2.join(sim1)

                self.assertEqual(1, len(sim1.joined_simulations))
                self.assertEqual(1, len(sim2.joined_simulations))
                self.assertIn(sim2, sim1.joined_simulations)
                self.assertIn(sim1, sim2.joined_simulations)
                self.assertEqual(sim1, sim2.joined_simulations[0])
                self.assertEqual(sim2, sim1.joined_simulations[0])

                # Once joined, joining should do nothing
                sim2.join(sim1)
                
                self.assertEqual(1, len(sim1.joined_simulations))
                self.assertEqual(1, len(sim2.joined_simulations))
                self.assertIn(sim2, sim1.joined_simulations)
                self.assertIn(sim1, sim2.joined_simulations)
                self.assertEqual(sim1, sim2.joined_simulations[0])
                self.assertEqual(sim2, sim1.joined_simulations[0])

                sim1.join(sim2)

                self.assertEqual(1, len(sim1.joined_simulations))
                self.assertEqual(1, len(sim2.joined_simulations))
                self.assertIn(sim2, sim1.joined_simulations)
                self.assertIn(sim1, sim2.joined_simulations)
                self.assertEqual(sim1, sim2.joined_simulations[0])
                self.assertEqual(sim2, sim1.joined_simulations[0])


        # Test some methods
        with sim1.archive.nosave():
            with sim2.archive.nosave():
                sim1.join(sim2)

                self.assertEqual('out000.sph', os.path.basename(sim1.get_start_file()))
                self.assertEqual('restartrad.sph.orig', os.path.basename(sim2.get_start_file()))
                self.assertEqual(sim1.get_output(), sim2.get_output())

                # Make sure both simulations correctly detect the same output
                # files in the same orders
                out1 = sim1.get_outputfiles()
                out2 = sim2.get_outputfiles()
                for i, name in enumerate(['out000.sph', 'out001.sph', 'out002.sph', 'out003.sph']):
                    self.assertEqual(os.path.basename(out1[i]), os.path.basename(out2[i]))
                    self.assertEqual(name, os.path.basename(out1[i]))
                    self.assertEqual(name, os.path.basename(out2[i]))


                # Test the include_joined keyword
                self.assertEqual(
                    'out002.sph',
                    os.path.basename(sim2.get_outputfiles(include_joined = False)[0]),
                )
                self.assertEqual(
                    'out000.sph',
                    os.path.basename(sim1.get_outputfiles(include_joined = False)[0]),
                )
        
        # I built the simulations to have identical log files
        #self.assertEqual(
        #    sim1.get_logfiles()[0],
        #    sim2.get_logfiles()[0],
        #)

        # Make sure to restore the original behavior
        #sim1.split()
        

if __name__ == "__main__":
    unittest.main(failfast=True)
