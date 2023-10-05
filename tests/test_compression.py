import unittest
import os
import starsmashertools

curdir = os.getcwd()
directory = os.path.join(curdir, 'data')
simulation = starsmashertools.get_simulation(directory)

files = ['log0.sph', 'out000.sph', 'sph.eos']
files = [os.path.join(directory, f) for f in files]

class TestCompression(unittest.TestCase):
    def testSerial(self):
        mtimes = {}
        for f in files:
            mtimes[f] = starsmashertools.helpers.path.getmtime(f)
        
        simulation.compress(parallel=False, verbose=False)
        self.assertFalse(os.path.isfile(os.path.join(directory, 'log0.sph')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'out000.sph')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'sph.eos')))
        self.assertTrue(os.path.isfile(os.path.join(directory, 'data.tar.gz')))
        
        simulation.decompress(parallel=False, verbose=False)
        self.assertTrue(os.path.isfile(os.path.join(directory, 'log0.sph')))
        self.assertTrue(os.path.isfile(os.path.join(directory, 'out000.sph')))
        self.assertTrue(os.path.isfile(os.path.join(directory, 'sph.eos')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'data.tar.gz')))

        for f, mtime in mtimes.items():
            self.assertEqual(mtime, starsmashertools.helpers.path.getmtime(f))
        
    def testParallel(self):
        mtimes = {}
        for f in files:
            mtimes[f] = starsmashertools.helpers.path.getmtime(f)
        
        simulation.compress(parallel=True, verbose=False)
        self.assertFalse(os.path.isfile(os.path.join(directory, 'log0.sph')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'out000.sph')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'sph.eos')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'log0.sph.gz')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'out000.sph.gz')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'sph.eos.gz')))
        self.assertTrue(os.path.isfile(os.path.join(directory, 'data.tar.gz')))

        simulation.decompress(parallel=True, verbose=False)
        self.assertTrue(os.path.isfile(os.path.join(directory, 'log0.sph')))
        self.assertTrue(os.path.isfile(os.path.join(directory, 'out000.sph')))
        self.assertTrue(os.path.isfile(os.path.join(directory, 'sph.eos')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'log0.sph.gz')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'out000.sph.gz')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'sph.eos.gz')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'data.tar.gz')))

        for f, mtime in mtimes.items():
            self.assertEqual(mtime, starsmashertools.helpers.path.getmtime(f))
            
if __name__ == "__main__":
    unittest.main(failfast=True)
