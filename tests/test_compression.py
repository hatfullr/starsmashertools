import unittest
import os
import starsmashertools

curdir = os.getcwd()
directory = os.path.join(curdir, 'data')
simulation = starsmashertools.get_simulation(directory)

class TestCompression(unittest.TestCase):
    def testFiles(self):
        expected = [
            'out000.sph',
            'log0.sph',
            'sph.eos',
            os.path.join('stuff','out000.sph'),
        ]
        expected = [os.path.join(directory, e) for e in expected]
        files = simulation._get_files_for_compression()
        self.assertEqual(len(expected), len(files))
        for f in expected:
            self.assertIn(f, files)
    
    def testCompress(self):
        simulation.compress(verbose=False)

    def testDecompress(self):
        simulation.decompress(verbose=False)
        
        
if __name__ == "__main__":
    unittest.main()
