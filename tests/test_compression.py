import unittest
import os
import starsmashertools
import gzip
import shutil
import time

def copydir(directory, new_directory):
    shutil.copytree(directory, new_directory)
    for filename in os.listdir(new_directory):
        path = os.path.join(new_directory, filename)
        orig = os.path.join(directory, filename)
        os.utime(path, times=(time.time(), os.path.getmtime(orig)))


class TestCompression(unittest.TestCase):
    def setUp(self):
        curdir = os.getcwd()
        self.orig_directory = os.path.join(curdir, 'data')

        self.orig_files = ['log0.sph', 'out000.sph', 'sph.eos']
        self.orig_files = [os.path.join(self.orig_directory, f) for f in self.orig_files]
        
        new_directory = self.orig_directory+"_test"

        if os.path.isdir(new_directory): shutil.rmtree(new_directory)
        
        copydir(self.orig_directory, new_directory)

        self.simulation = starsmashertools.get_simulation(new_directory)

        self.new_files = []
        for _file in self.orig_files:
            basename = starsmashertools.helpers.path.basename(_file)
            self.new_files += [starsmashertools.helpers.path.join(self.simulation.directory, basename)]
        
        

    def tearDown(self):
        for orig_file, new_file in zip(self.orig_files, self.new_files):
            orig_mtime = starsmashertools.helpers.path.getmtime(orig_file)
            if starsmashertools.helpers.path.isfile(new_file):
                new_mtime = starsmashertools.helpers.path.getmtime(new_file)
                self.assertEqual(orig_mtime, new_mtime)
        
        self.assertFalse(os.path.isfile(
            starsmashertools.helpers.path.join(self.simulation.directory, 'compression.sstools')
        ))

        shutil.rmtree(self.simulation.directory)
    
    def testCompressSerial(self):
        self.simulation.compress(parallel=False, verbose=False)
        for _file in self.new_files:
            self.assertFalse(os.path.isfile(_file))
        fname = self.simulation._get_compression_filename()
        self.assertTrue(os.path.isfile(fname))
        
                         

    def testDecompressSerial(self):
        self.simulation.compress(parallel=False, verbose=False)
        self.simulation.decompress(parallel=False, verbose=False)
        fname = self.simulation._get_compression_filename()
        self.assertFalse(os.path.isfile(fname))
        
    """
    def testParallel(self):
        self.simulation.compress(parallel=True, verbose=False)
        self.assertFalse(os.path.isfile(os.path.join(directory, 'log0.sph')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'out000.sph')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'sph.eos')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'log0.sph.gz')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'out000.sph.gz')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'sph.eos.gz')))
        self.assertTrue(os.path.isfile(os.path.join(directory, 'data.tar.gz')))

        self.simulation.decompress(parallel=True, verbose=False)
        self.assertTrue(os.path.isfile(os.path.join(directory, 'log0.sph')))
        self.assertTrue(os.path.isfile(os.path.join(directory, 'out000.sph')))
        self.assertTrue(os.path.isfile(os.path.join(directory, 'sph.eos')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'log0.sph.gz')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'out000.sph.gz')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'sph.eos.gz')))
        self.assertFalse(os.path.isfile(os.path.join(directory, 'data.tar.gz')))
    
    def testCompressInterruptedParallel(self):
        dname1 = starsmashertools.helpers.path.join(curdir, 'test_half_compressed')
        dname2 = starsmashertools.helpers.path.join(curdir, 'data_half_compressed')
        if starsmashertools.helpers.path.isdir(dname1):
            shutil.rmtree(dname1)
        shutil.copytree(dname2, dname1)
        _simulation = starsmashertools.get_simulation(dname1)

        _simulation.compress(parallel=True, verbose=False)

        
        
        #shutil.rmtree(dname1)
    """
            
if __name__ == "__main__":
    unittest.main(failfast=True)
