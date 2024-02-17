import starsmashertools.helpers.file
import unittest
import os
import basetest

curdir = os.getcwd()

class TestFile(basetest.BaseTest):
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
        path = os.path.realpath(os.path.join('data', 'testfile'))

        # When we open the file, we should create a corresponding *.lock
        # file.
        t0 = time.time()
        with starsmashertools.helpers.file.open(
                path, 'r',
                timeout = 1.,
        ) as f:
            lock = starsmashertools.helpers.file.Lock.get(path)
            self.assertTrue(lock.locked)
            
            t0 = time.time()
            with starsmashertools.helpers.file.open(
                    path, 'r',
                    timeout = 1.e-3,
            ) as f2:
                pass
            self.assertAlmostEqual(time.time() - t0, 1.e-3, places=3)
        
if __name__ == "__main__":
    unittest.main(failfast=True)

