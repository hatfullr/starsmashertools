import starsmashertools
import unittest
import os
import time
import numpy as np
import basetest

curdir = os.getcwd()
simdir = os.path.join(curdir, 'data')

class TestLogFile(basetest.BaseTest):
    def setUp(self):
        self.simulation = starsmashertools.get_simulation(simdir)
        self.logfile = self.simulation.get_logfiles()[0]
    
    def test_get(self):
        self.assertEqual(int(self.logfile.get('nkernel is').strip()), self.simulation['nkernel'])
    def test_has_output_files(self):
        outputfiles = self.simulation.get_outputfiles()
        result = [True for o in outputfiles]
        self.assertEqual(self.logfile.has_output_files(outputfiles), result)

    def test_get_stop_time(self):
        self.assertEqual(19.99629, self.logfile.get_stop_time())

    def test_get_start_time(self):
        self.assertEqual(0, self.logfile.get_start_time())

    def test_get_iterations(self):
        iterations = list(self.logfile.get_iterations(0, 1))
        self.assertEqual(len(iterations), 2)
        self.assertEqual(iterations[0]['iteration'], 0)
        self.assertEqual(iterations[1]['iteration'], 1)

        iterations = list(self.logfile.get_iterations())
        self.assertEqual(len(iterations), 359)
        self.assertEqual(iterations[-1]['iteration'], 358)

    def test_dts(self):
        iterations = list(self.logfile.get_iterations())
        dts1 = iterations[0]['dts']
        dts2 = iterations[-1]['dts']

        arr1 = np.array([0.187, 0.562E-01, 0.100E+31, 0.100E+31, 0.100E+31, 0.100E+31, 0.458E-01])
        arr2 = np.array([0.107, 0.151, 0.647, 0.100E+31, 0.100E+31, 0.100E+31, 0.785E-01])
        for arr, dts in [(arr1, dts1), (arr2, dts2)]:
            for a,b in zip(dts, arr):
                self.assertEqual(a, b, msg = "Failed on dts = "+str(dts)+" logfile = "+str(self.logfile.path))

    

if __name__ == "__main__":
    unittest.main(failfast=True)
