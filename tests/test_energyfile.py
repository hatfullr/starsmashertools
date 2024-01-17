import unittest
import basetest
import numpy as np
import starsmashertools
import starsmashertools.lib.energyfile
import os
import numpy as np

curdir = os.getcwd()

class TestEnergyFile(basetest.BaseTest):
    def setUp(self):
        path = os.path.join(curdir, 'data')
        self.simulation = starsmashertools.get_simulation(path)
        
    def testBasic(self):
        self.simulation.get_energy(sort='t')
        energyfile = self.simulation.get_energyfiles(skip_rows = 1)[0]
        
        data = []
        with open(energyfile.path, 'r') as f:
            for line in f:
                data += [line.strip().split()]
        data = np.asarray(data, dtype=object).astype(float)

        for i, (key, val) in enumerate(energyfile.items()):
            self.assertTrue(np.array_equal(data[:,i], val))
        
if __name__ == '__main__':
    unittest.main(failfast=True)
