import starsmashertools.helpers.file
import unittest
import os
import basetest

curdir = os.getcwd()

class TestFile(basetest.BaseTest):

    def setUp(self):
        import starsmashertools
        import os
        
        for filename in os.listdir(starsmashertools.LOCK_DIRECTORY):
            path = os.path.join(starsmashertools.LOCK_DIRECTORY, filename)
            os.remove(path)
    
    def tearDown(self):
        for name in ['lock_test_file', 'lock_test_file2']:
            if os.path.isfile(name): os.remove(name)
            
    
    def test_get_phrase(self):
        result = starsmashertools.helpers.file.get_phrase(
            os.path.join('data', 'testfile'),
            'test',
        )
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], '4')
        self.assertEqual(result[1], '9')
        self.assertEqual(result[2], '')
        self.assertEqual(result[3], '56789')


        result = starsmashertools.helpers.file.get_phrase(
            os.path.join('data', 'testfile1'),
            'test',
        )
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], '4')
        self.assertEqual(result[1], '901234M6789012345')
        self.assertEqual(result[2], '01')
        self.assertEqual(result[3], '6789')

    def test_reversed_textfile(self):
        expected = """0123456789
test4test9
01234M6789
012345test
0test56789"""
        expected = expected.split("\n")
        
        f = starsmashertools.helpers.file.reverse_readline(
                os.path.join('data', 'testfile'),
        )
        
        for i, line in enumerate(f):
            if line:
                self.assertEqual(expected[len(expected) - i - 1], line)

    def test_lock(self):
        import starsmashertools.helpers.file
        import starsmashertools
        import os
        
        lockdir = starsmashertools.LOCK_DIRECTORY
        
        # Touch a new file for us to use
        filename = 'lock_test_file'
        open(filename, 'x').close()

        self.assertEqual(0, len(os.listdir(lockdir)), msg="Clear lock directory first")

        all_modes = starsmashertools.helpers.file.all_modes
        
        # All modes should register file locking
        for mode in all_modes:
            if mode == 'x': continue
            with starsmashertools.helpers.file.open(filename, mode) as f:
                self.assertEqual(1, len(os.listdir(lockdir)), msg = "Mode '%s' didn't cause locking" % mode)

            self.assertEqual(0, len(os.listdir(lockdir)), msg = "Mode '%s' left a residual lock file" % mode)

        # Try nested open statements
        readonly_modes = starsmashertools.helpers.file.modes['readonly']
        write_modes = starsmashertools.helpers.file.modes['write']
        for mode1 in all_modes:
            if mode1 == 'x': continue
            with starsmashertools.helpers.file.open(filename, mode1) as f1:
                for mode2 in all_modes:
                    if mode2 == 'x': continue
                    try:
                        with starsmashertools.helpers.file.open(filename, mode2, timeout=1.e-6) as f2:
                            if mode1 in write_modes and mode2 in write_modes:
                                self.assertEqual(
                                    1, len(os.listdir(lockdir)),
                                    msg = "One lock file should be created: '%s', '%s'" % (mode1, mode2)
                                )
                            else:
                                self.assertEqual(
                                    2, len(os.listdir(lockdir)),
                                    msg = "Two lock files should be created: '%s', '%s'" % (mode1, mode2)
                                )
                    except TimeoutError as e:
                        if (mode2 in write_modes or mode1 in write_modes):
                            with self.assertRaises(TimeoutError):
                                raise
                        else:
                            raise Exception("Failed timeout test: '%s', '%s'" % (mode1, mode2)) from e
                    self.assertEqual(
                        1, len(os.listdir(lockdir)),
                        msg = "Nested open statements leave residual lock files: '%s', '%s'" % (mode1, mode2)
                    )

    def test_is_locked(self):
        import starsmashertools.helpers.file
        import starsmashertools
        import os
        import time
        
        lockdir = starsmashertools.LOCK_DIRECTORY
        
        # Touch a new file for us to use
        filename = 'lock_test_file'
        open(filename, 'x').close()

        self.assertEqual(0, len(os.listdir(lockdir)), msg="Clear lock directory first")

        for mode in starsmashertools.helpers.file.all_modes:
            if mode == 'x': continue
            self.assertFalse(starsmashertools.helpers.file.is_locked(filename))
            
            with starsmashertools.helpers.file.open(filename, mode) as f:
                self.assertTrue(starsmashertools.helpers.file.is_locked(filename))

            self.assertFalse(starsmashertools.helpers.file.is_locked(filename))


    def test_get_lock_files(self):
        import starsmashertools.helpers.file
        import starsmashertools
        import os
        import time
        
        lockdir = starsmashertools.LOCK_DIRECTORY
        
        # Touch a new file for us to use
        filename = 'lock_test_file'
        open(filename, 'x').close()

        self.assertEqual(0, len(os.listdir(lockdir)), msg="Clear lock directory first")

        for mode in starsmashertools.helpers.file.all_modes:
            if mode == 'x': continue
            lockfiles = list(starsmashertools.helpers.file.get_lock_files(filename))
            self.assertEqual(0, len(lockfiles))

            with starsmashertools.helpers.file.open(filename, mode) as f:
                lockfiles = list(starsmashertools.helpers.file.get_lock_files(filename))
                self.assertEqual(1, len(lockfiles))

                for _mode in starsmashertools.helpers.file.all_modes:
                    if _mode in [mode, 'x']: continue
                    with starsmashertools.helpers.file.open(filename, _mode, timeout=0) as f2:
                        lockfiles = list(starsmashertools.helpers.file.get_lock_files(filename))
                        self.assertEqual(2, len(lockfiles))
                    lockfiles = list(starsmashertools.helpers.file.get_lock_files(filename))
                    self.assertEqual(1, len(lockfiles))
            lockfiles = list(starsmashertools.helpers.file.get_lock_files(filename))
            self.assertEqual(0, len(lockfiles))

    def test_unlock(self):
        import starsmashertools.helpers.file
        import starsmashertools
        import os
        import time
        
        lockdir = starsmashertools.LOCK_DIRECTORY
        
        # Touch a new file for us to use
        filename = 'lock_test_file'
        open(filename, 'x').close()

        self.assertEqual(0, len(os.listdir(lockdir)), msg="Clear lock directory first")
        
        with starsmashertools.helpers.file.open(filename, 'r') as f:
            self.assertTrue(starsmashertools.helpers.file.is_locked(filename))
            starsmashertools.helpers.file.unlock(filename)
            self.assertFalse(starsmashertools.helpers.file.is_locked(filename))
        self.assertFalse(starsmashertools.helpers.file.is_locked(filename))

    def test_unlock_all(self):
        import starsmashertools.helpers.file
        import starsmashertools
        import os
        import time
        
        lockdir = starsmashertools.LOCK_DIRECTORY
        
        # Touch a new file for us to use
        filename = 'lock_test_file'
        open(filename, 'x').close()
        filename2 = 'lock_test_file2'
        open(filename2, 'x').close()
        
        self.assertEqual(0, len(os.listdir(lockdir)), msg="Clear lock directory first")

        with starsmashertools.helpers.file.open(filename, 'r') as f:
            self.assertTrue(starsmashertools.helpers.file.is_locked(filename))
            with starsmashertools.helpers.file.open(filename2, 'r') as f2:
                self.assertTrue(starsmashertools.helpers.file.is_locked(filename2))
                starsmashertools.helpers.file.unlock_all()

                self.assertFalse(starsmashertools.helpers.file.is_locked(filename))
                self.assertFalse(starsmashertools.helpers.file.is_locked(filename2))

        self.assertFalse(starsmashertools.helpers.file.is_locked(filename))
        self.assertFalse(starsmashertools.helpers.file.is_locked(filename2))
    
    def test_parallel_lock(self):
        import multiprocessing
        import starsmashertools
        import time
        import os

        # Touch a new file for us to use
        filename = 'lock_test_file'
        open(filename, 'x').close()
        
        def check_lock(mode, output_queue):
            import starsmashertools.helpers.file
            import os
            try:
                with starsmashertools.helpers.file.open(
                        filename, mode, timeout=1.e-6,
                ) as f:
                    output_queue.put([mode, False]) # was not locked
            except TimeoutError:
                output_queue.put([mode, True]) # was locked
        
        modes = starsmashertools.helpers.file.modes
        all_modes = []
        for key, val in modes.items():
            all_modes += val
        
        manager = multiprocessing.Manager()
        output_queue = manager.Queue()
        
        nprocs = multiprocessing.cpu_count()

        for mode1 in all_modes:
            if mode1 == 'x': continue
            with starsmashertools.helpers.file.open(
                    filename, mode1,
            ) as f:
                for mode2 in all_modes:
                    #print(mode1, mode2)
                    if mode2 == 'x': continue
                    processes = []
                    for i in range(nprocs):
                        processes += [multiprocessing.Process(
                            target = check_lock,
                            args = (mode2, output_queue,),
                            daemon = True,
                        )]

                    for process in processes: process.start()

                    for process in processes:
                        process.join()
                        process.terminate()
                
                    # Make sure all the processes found the right results
                
                    expected = None
                    if mode1 in modes['readonly'] and mode2 in modes['readonly']:
                        expected = False
                    elif mode1 in modes['readonly'] and mode2 in modes['write']:
                        expected = True
                    elif mode1 in modes['write'] and mode2 in modes['readonly']:
                        expected = True
                    elif mode1 in modes['write'] and mode2 in modes['write']:
                        expected = True
                    else:
                        raise Exception("Unknown modes: '%s', '%s'" % (mode1,mode2))
                    
                    while not output_queue.empty():
                        _mode, locked = output_queue.get()
                        self.assertEqual(_mode, mode2, msg="process got wrong mode")
                        self.assertEqual(expected, locked, msg = "Mode failed: '%s', '%s'" % (mode1, mode2))

if __name__ == "__main__":
    unittest.main(failfast=True)

