import starsmashertools
import unittest
import os
import time
import numpy as np
import basetest

curdir = os.getcwd()
simdir = os.path.join(curdir, 'data')
simulation = starsmashertools.get_simulation(simdir)
logfile = simulation.get_logfiles()[0]

class TestLogFile(basetest.BaseTest):
    def test_get(self):
        self.assertEqual(int(logfile.get('nkernel is').strip()), simulation['nkernel'])
    def test_has_output_files(self):
        outputfiles = simulation.get_outputfiles()
        result = [True for o in outputfiles]
        self.assertEqual(logfile.has_output_files(outputfiles), result)

    def test_get_stop_time(self):
        self.assertEqual(9.97473, logfile.get_stop_time())

    def test_get_start_time(self):
        self.assertEqual(0, logfile.get_start_time())

    def test_get_iterations(self):
        iterations = logfile.get_iterations(0, 1)
        self.assertEqual(len(iterations), 2)
        self.assertEqual(iterations[0]['iteration'], 0)
        self.assertEqual(iterations[1]['iteration'], 1)

        iterations = logfile.get_iterations()
        self.assertEqual(len(iterations), 224)
        self.assertEqual(iterations[-1]['iteration'], 223)

    def test_dts(self):
        iterations = logfile.get_iterations()
        dts1 = iterations[0]['dts']
        dts2 = iterations[-1]['dts']

        arr1 = np.array([0.187, 0.562E-01, 0.100E+31, 0.100E+31, 0.100E+31, 0.100E+31, 0.458E-01])
        arr2 = np.array([0.106, 0.813E-01, 0.319, 0.100E+31, 0.100E+31, 0.100E+31, 0.580E-01])
        for arr, dts in [(arr1, dts1), (arr2, dts2)]:
            for a,b in zip(dts, arr):
                self.assertEqual(a, b, msg = "Failed on dts = "+str(dts)+" logfile = "+str(logfile.path))

    

if __name__ == "__main__":
    unittest.main(failfast=True)
