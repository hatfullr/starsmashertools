import starsmashertools
import unittest
import os
import time

curdir = os.getcwd()
simdir = os.path.join(curdir, 'data')
simulation = starsmashertools.get_simulation(simdir)
logfile = simulation.get_logfiles()[0]

class TestLogFile(unittest.TestCase):
    def testGet(self):
        self.assertEqual(int(logfile.get('nkernel is').strip()), simulation['nkernel'])
    def testHasOutputFiles(self):
        outputfiles = simulation.get_outputfiles()
        result = [True for o in outputfiles]
        self.assertEqual(logfile.has_output_files(outputfiles), result)

    def testGetStopTime(self):
        self.assertEqual(1000.04018, logfile.get_stop_time())

    def testGetStartTime(self):
        self.assertEqual(0, logfile.get_start_time())

if __name__ == "__main__":
    unittest.main(failfast=True)
