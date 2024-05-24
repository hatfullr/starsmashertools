import atexit
import os
test_directory = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'data_test',
)
def remove_test_directory():
    if os.path.isdir(test_directory):
        shutil.rmtree(test_directory)
atexit.register(remove_test_directory)


import unittest
import starsmashertools
import shutil
import time
import multiprocessing
import zipfile
import copy
import basetest


def copydir(directory, new_directory):
    import starsmashertools.helpers.path
    shutil.copytree(directory, new_directory)
    for filename in os.listdir(new_directory):
        path = os.path.join(new_directory, filename)
        orig = os.path.join(directory, filename)
        os.utime(path, times=(
            int(time.time()),
            starsmashertools.helpers.path.getmtime(orig),
        ))

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size




class TestCompression(basetest.BaseTest):
    def __init__(self, *args, **kwargs):
        super(TestCompression, self).__init__(*args, **kwargs)
    
    def setUp(self):
        orig_directory = os.path.join(
            starsmashertools.SOURCE_DIRECTORY,
            'tests',
            'data',
        )

        self.orig_files = ['log0.sph', 'out000.sph', 'sph.eos']
        self.orig_files = [os.path.join(orig_directory, f) for f in self.orig_files]
        
        remove_test_directory()
        
        copydir(orig_directory, test_directory)

        self.simulation = starsmashertools.get_simulation(test_directory)

        self.simulation_initial_compression = copy.deepcopy(self.simulation.compressed)
        
        self.new_files = []
        for _file in self.orig_files:
            basename = starsmashertools.helpers.path.basename(_file)
            self.new_files += [starsmashertools.helpers.path.join(self.simulation.directory, basename)]
    
    def tearDown(self):
        for orig_file, new_file in zip(self.orig_files, self.new_files):
            orig_mtime = starsmashertools.helpers.path.getmtime(orig_file)
            if starsmashertools.helpers.path.isfile(new_file):
                new_mtime = starsmashertools.helpers.path.getmtime(new_file)
                # We have to allow for a small bit of imprecision here, because
                # zipfile is incapable of keeping timestamps to enough precision
                # to satisfy an assertEqual call.
                self.assertAlmostEqual(
                    int(orig_mtime) * 0.1,
                    int(new_mtime) * 0.1,
                    msg=new_file, places=0,
                )
        
        compression_filename = self.simulation._get_compression_filename() + ".json"
        self.assertFalse(os.path.isfile(compression_filename))
        remove_test_directory()
    
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
        if __name__ == "__main__":
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
        if __name__ == "__main__":
            print("parallel=",total_compressed, total_regular)

        self.simulation.decompress(verbose = False)
        self.assertFalse(self.simulation.compressed)
        fname = self.simulation._get_compression_filename()
        self.assertFalse(os.path.isfile(fname))
    #"""
if __name__ == "__main__":
    unittest.main(failfast=True)
