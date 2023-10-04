import unittest
import os
import starsmashertools

curdir = os.getcwd()
directory = os.path.join(curdir, 'data')
simulation = starsmashertools.get_simulation(directory)

class TestCompression(unittest.TestCase):
    def testCompress(self):
        simulation.compress(verbose=False)

    def testDecompress(self):
        simulation.decompress(verbose=False)
        
        
if __name__ == "__main__":
    unittest.main(failfast=True)
