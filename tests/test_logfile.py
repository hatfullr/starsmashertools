import starsmashertools
import unittest
import os
import time
import numpy as np

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

    def testGetdts(self):
        dts = logfile.get_dts()
        arr1 = np.array([0.187, 0.562E-01, 0.100E+31, 0.100E+31, 0.100E+31, 0.100E+31, 0.458E-01])
        arr2 = np.array([0.110, 1.76, 764., 0.100E+31, 0.100E+31, 0.100E+31, 0.105])
        for a,b in zip(dts[0], arr1):
            self.assertEqual(a, b)
        for a,b in zip(dts[-1], arr2):
            self.assertEqual(a, b)

    def testGetIterations(self):
        iterations = logfile.get_iterations()
        #print(iterations)

if __name__ == "__main__":
    unittest.main(failfast=True)
