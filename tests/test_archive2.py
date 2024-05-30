import unittest
import starsmashertools.lib.archive2
import os
import struct
import starsmashertools.helpers.pickler

class Test(unittest.TestCase):
    filename = 'test'
    def setUp(self):
        if os.path.exists(Test.filename): os.remove(Test.filename)

    def tearDown(self):
        if os.path.exists(Test.filename): os.remove(Test.filename)
        
    def test_simple(self):
        a = starsmashertools.lib.archive2.Archive(Test.filename)
        with self.assertRaises(FileNotFoundError):
            a.get('test')

        a.set('test', 'simple')
        self.assertEqual(len(a._footer), 1)
        self.assertEqual(list(a._footer.values())[-1].identifier, 'test')
        self.assertEqual(list(a._footer.values())[-1].start, 0)
        self.assertEqual(list(a._footer.values())[-1].stop, 28)

        with open('test', 'rb') as f:
            f.seek(-4, 2)
            self.assertEqual(12, struct.unpack('<i', f.read(4))[0])

        self.assertEqual(b'gASVCgAAAAAAAACMBnNpbXBsZZQu', a.get('test', raw = True))
        
        self.assertEqual(a.get('test'), 'simple')


    def test_overwrite_no_truncate(self):
        a = starsmashertools.lib.archive2.Archive(Test.filename)
        a.set('test', 'a')
        a.set('test2', 'b')

        previous_start_stop = (a._footer['test'].start, a._footer['test'].stop)
        a.set('test', 'c')
        self.assertEqual('c', a.get('test'))
        self.assertEqual(starsmashertools.helpers.pickler.pickle_object('c'), a.get('test', raw = True))
        self.assertEqual(previous_start_stop, (a._footer['test'].start, a._footer['test'].stop))
        self.assertEqual('b', a.get('test2'))

    def test_overwrite_truncate(self):
        a = starsmashertools.lib.archive2.Archive(Test.filename)
        a.set('test', 'abcd')
        a.set('test2', 'b')

        previous_start_stop = (a._footer['test'].start, a._footer['test'].stop)
        a.set('test', 'c')
        self.assertEqual('c', a.get('test'))
        self.assertEqual(starsmashertools.helpers.pickler.pickle_object('c'), a.get('test', raw = True))
        self.assertNotEqual(previous_start_stop, (a._footer['test'].start, a._footer['test'].stop))
        self.assertEqual('b', a.get('test2'))

        # Try the truncation type which extends the data space
        a.set('test', 'abcd')
        self.assertEqual('abcd', a.get('test'))
        self.assertEqual('b', a.get('test2'))

    def test_big_file(self):
        import numpy as np
        data = np.linspace(0, 1, int(1e7))
        a = starsmashertools.lib.archive2.Archive(Test.filename)
        a.set('test', data)
        self.assertTrue(np.array_equal(a.get('test'), data))

    def test_compression(self):
        a = starsmashertools.lib.archive2.Archive(Test.filename)
        a.set('test', [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0])
        with self.assertRaises(starsmashertools.lib.archive2.CompressedError):
            a.decompress()
        previous_size = os.path.getsize(a.path)
        a.compress()
        self.assertLess(previous_size, os.path.getsize(a.path))

        with self.assertRaises(starsmashertools.lib.archive2.CompressedError):
            a.compress()



if __name__ == '__main__':
    unittest.main(failfast = True)
