import starsmashertools.helpers.file
import unittest
import os
import basetest

curdir = os.getcwd()

class TestFile(basetest.BaseTest):

    def tearDown(self):
        if os.path.isfile('lock_test_file'):
            os.remove('lock_test_file')
    
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
        import time
        import starsmashertools.helpers.file
        import os

        # Touch a new file for us to use
        open('lock_test_file', 'w').close()
        
        path = 'lock_test_file'

        # When we open the file, we should create a corresponding *.lock
        # file.
        t0 = time.time()
        with starsmashertools.helpers.file.open(
                path, 'r',
                timeout = 1.,
                lock = False,
        ) as f:
            self.assertFalse(starsmashertools.helpers.file.Lock.locked(path))
            self.assertEqual(0, len(starsmashertools.helpers.file.Lock.get_all_lockfiles(path)))
            
            with starsmashertools.helpers.file.open(
                    path, 'a',
                    lock = True,
            ) as f2:
                self.assertTrue(starsmashertools.helpers.file.Lock.locked(path))

            self.assertFalse(starsmashertools.helpers.file.Lock.locked(path))

        with starsmashertools.helpers.file.open(
                path, 'w',
        ) as f:
            
            self.assertTrue(starsmashertools.helpers.file.Lock.locked(path))

            t0 = time.time()
            with starsmashertools.helpers.file.open(
                    path, 'r',
                    timeout = 1.e-2,
            ) as f2:
                timer = time.time() - t0
            
            # Reasonable precision. Doesn't matter a whole lot.
            self.assertLessEqual(timer, 1.e-2 + 1.e-3)


    def test_parallel(self):
        import multiprocessing

        writables = ['w', 'a', 'w+', 'r+', 'wb', 'wb+', 'rb+', 'ab']
        all_modes = ['r', 'rb'] + writables

        # Touch a new file for us to use
        open('lock_test_file', 'w').close()
        
        def check_lock(output_queue):
            import starsmashertools.helpers.file
            output_queue.put(starsmashertools.helpers.file.Lock.locked('lock_test_file'))
        
        manager = multiprocessing.Manager()
        output_queue = manager.Queue()
        
        nprocs = 4

        for mode in all_modes:
            processes = []
            for i in range(nprocs):
                processes += [multiprocessing.Process(
                    target = check_lock,
                    args = (output_queue, ),
                    daemon = True,
                )]

            with starsmashertools.helpers.file.open(
                    'lock_test_file', mode,
                    lock = True,
            ) as f:
                self.assertTrue(starsmashertools.helpers.file.Lock.locked('lock_test_file'), msg="File never got locked")
                for process in processes: process.start()

                for process in processes:
                    process.join()
                    process.terminate()

            while not output_queue.empty():
                self.assertTrue(output_queue.get(), msg = "Mode '%s' failed" % mode)

        # Writable modes should always lock
        
        for mode in all_modes:
            processes = []
            for i in range(nprocs):
                processes += [multiprocessing.Process(
                    target = check_lock,
                    args = (output_queue, ),
                    daemon = True,
                )]

            with starsmashertools.helpers.file.open(
                    'lock_test_file', mode,
                    lock = False,
            ) as f:
                locked = starsmashertools.helpers.file.Lock.locked('lock_test_file')
                if mode in writables:
                    self.assertTrue(locked, msg="File was not locked")
                else:
                    self.assertFalse(locked, msg="File was locked")
                
                for process in processes: process.start()

                for process in processes:
                    process.join()
                    process.terminate()

            while not output_queue.empty():
                if mode in writables:
                    self.assertTrue(output_queue.get(), msg = "Mode '%s' failed" % mode)
                else: self.assertFalse(output_queue.get(), msg = "Mode '%s' failed" % mode)
        
        
if __name__ == "__main__":
    unittest.main(failfast=True)

