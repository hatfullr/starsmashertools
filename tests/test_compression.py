import unittest
import os
import starsmashertools
import shutil
import time
import multiprocessing
import zipfile
import copy

def copydir(directory, new_directory):
    shutil.copytree(directory, new_directory)
    for filename in os.listdir(new_directory):
        path = os.path.join(new_directory, filename)
        orig = os.path.join(directory, filename)
        os.utime(path, times=(time.time(), os.path.getmtime(orig)))

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


class TestCompression(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCompression, self).__init__(*args, **kwargs)
        self.skip_tearDown = False
    
    def setUp(self):
        curdir = os.getcwd()
        self.orig_directory = os.path.join(curdir, 'data')

        self.orig_files = ['log0.sph', 'out000.sph', 'sph.eos']
        self.orig_files = [os.path.join(self.orig_directory, f) for f in self.orig_files]
        
        new_directory = self.orig_directory+"_test"

        if os.path.isdir(new_directory): shutil.rmtree(new_directory)
        
        copydir(self.orig_directory, new_directory)

        self.simulation = starsmashertools.get_simulation(new_directory)

        self.simulation_initial_compression = copy.deepcopy(self.simulation.compressed)
        
        self.new_files = []
        for _file in self.orig_files:
            basename = starsmashertools.helpers.path.basename(_file)
            self.new_files += [starsmashertools.helpers.path.join(self.simulation.directory, basename)]
        
        

    def tearDown(self):
        if self.skip_tearDown: return

        if self.simulation.compressed != self.simulation_initial_compression:
            self.assertAlmostEqual(self.simulation.compression_progress, 1)
        
        for orig_file, new_file in zip(self.orig_files, self.new_files):
            orig_mtime = starsmashertools.helpers.path.getmtime(orig_file)
            if starsmashertools.helpers.path.isfile(new_file):
                new_mtime = starsmashertools.helpers.path.getmtime(new_file)
                self.assertEqual(orig_mtime, new_mtime)
        
        compression_filename = self.simulation._get_compression_filename() + ".json"
        self.assertFalse(os.path.isfile(compression_filename))
        
        shutil.rmtree(self.simulation.directory)
    
    def testSerial(self):
        total_regular = get_size(start_path = self.simulation.directory)
        self.simulation.compress(nprocs=1, verbose=False)
        self.assertTrue(self.simulation.compressed)
        
        for _file in self.new_files:
            self.assertFalse(os.path.isfile(_file))
        fname = self.simulation._get_compression_filename()
        self.assertTrue(os.path.isfile(fname))

        total_compressed = get_size(start_path = self.simulation.directory)
        self.assertLess(total_compressed, total_regular)
        print("serial=",total_compressed, total_regular)

        self.simulation.decompress(verbose=False)
        
        self.assertFalse(self.simulation.compressed)
        fname = self.simulation._get_compression_filename()
        self.assertFalse(os.path.isfile(fname))
    
    def testParallel(self):
        total_regular = get_size(start_path = self.simulation.directory)
        self.simulation.compress(nprocs=0, verbose=False)
        self.assertTrue(self.simulation.compressed)

        for _file in self.new_files:
            self.assertFalse(os.path.isfile(_file))
        fname = self.simulation._get_compression_filename()
        self.assertTrue(os.path.isfile(fname))

        directory = self.simulation.directory
        basename = starsmashertools.helpers.path.basename(directory)
        for i in range(multiprocessing.cpu_count()):
            fname = starsmashertools.helpers.path.join(directory, basename+"."+str(i))
            self.assertFalse(os.path.isfile(fname))


        total_compressed = get_size(start_path = self.simulation.directory)
        self.assertLess(total_compressed, total_regular)
        print("parallel=",total_compressed, total_regular)

        self.simulation.decompress(verbose = False)
        self.assertFalse(self.simulation.compressed)
        fname = self.simulation._get_compression_filename()
        self.assertFalse(os.path.isfile(fname))
    #"""
if __name__ == "__main__":
    unittest.main(failfast=True)
