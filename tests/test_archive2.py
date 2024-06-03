import unittest
import starsmashertools.lib.archive2
import os
import struct
import pickle
import time
import io

class Test(unittest.TestCase):
    filename = 'test'
    def setUp(self):
        if os.path.exists(Test.filename): os.remove(Test.filename)
        self.archive = starsmashertools.lib.archive2.Archive(Test.filename)

    def tearDown(self):
        if os.path.exists(Test.filename): os.remove(Test.filename)
        if os.path.exists('testarchiveadd'): os.remove('testarchiveadd')
        
    def test_simple(self):
        with self.assertRaises(KeyError):
            self.archive['test']
        
        self.archive['test'] = 'simple'
        self.assertEqual(self.archive['test'], 'simple')
        
        self.archive['test'] = 'simple'
        
        self.archive['test2'] = 'hi'
        
        self.archive['test'] = 'simple2'
        self.assertEqual(self.archive['test'], 'simple2')
        
        self.archive['test'] = 'simp'
        self.assertEqual(self.archive['test'], 'simp')

    def test_setitem(self):
        previous_size = self.archive.size()
        self.archive['test'] = 'simple'
        expected = pickle.dumps('simple')
        self.assertEqual(expected, self.archive._buffer[:len(expected)])
        self.assertLess(previous_size, self.archive.size())

    def test_delitem(self):
        self.archive['test'] = 'hello'
        del self.archive['test']
        self.assertNotIn('test', self.archive)
        with self.assertRaises(KeyError):
            self.archive['test']

    def test_contains(self):
        self.archive['test'] = 'hello'
        self.assertIn('test', self.archive)

    def test_keys(self):
        self.archive['0'] = 0
        self.archive['1'] = 1
        self.archive['2'] = 2
        exp = ['0', '1', '2']
        self.assertEqual(len(exp), len(self.archive.keys()))
        for expected, found in zip(exp, self.archive.keys()):
            self.assertEqual(expected, found)

    def test_values(self):
        self.archive['0'] = 0
        self.archive['1'] = 1
        self.archive['2'] = 2
        for i, value in enumerate(self.archive.values()):
            self.assertEqual(i, value)

    def test_items(self):
        self.archive['0'] = 0
        self.archive['1'] = 1
        self.archive['2'] = 2
        for i, (key, val) in enumerate(self.archive.items()):
            self.assertEqual(str(i), key)
            self.assertEqual(i, val)

    def test_size(self):
        self.assertEqual(0, self.archive.size())
        self.archive['0'] = 0
        self.assertEqual(self.archive._buffer.size(), self.archive.size())

    def test_readonly(self):
        archive = starsmashertools.lib.archive2.Archive(
            Test.filename,
            readonly = True,
        )
        with self.assertRaises(io.UnsupportedOperation):
            archive['hello'] = 'hi'

    def test_add(self):
        self.archive['test'] = 'hi'
        self.archive.add('test', 0)
        self.assertEqual(0, self.archive['test'])

        time.sleep(1.e-4)
        open('testarchiveadd', 'x').close()
        self.archive.add(
            'test',
            0,
            origin = 'testarchiveadd',
            replace = 'mtime',
        )
        self.assertEqual(0, self.archive['test'])

        self.archive['test'] = 'hi'
        self.archive.add(
            'test',
            0,
            origin = 'testarchiveadd',
            replace = 'mtime',
        )
        self.assertEqual('hi', self.archive['test'])

    def test_complex(self):
        import starsmashertools.helpers.nesteddict

        d = starsmashertools.helpers.nesteddict.NestedDict({
            'level 1' : {
                'level 2' : {
                    'test' : [i for i in range(4)],
                },
            },
        })
        
        self.archive["('level 1', 'level 2', 'test')"] = [i for i in range(4, 8)]
        for branch, leaf in d.flowers():
            key = str(branch)
            self.archive[key] += leaf
        
        self.assertEqual(1, len(self.archive))
        self.assertEqual(
            self.archive["('level 1', 'level 2', 'test')"],
            [4, 5, 6, 7, 0, 1, 2, 3],
        )

        for branch, leaf in d.flowers():
            key = str(branch)
            self.archive[key] += leaf

        self.assertEqual(1, len(self.archive))
        self.assertEqual(
            self.archive["('level 1', 'level 2', 'test')"],
            [4, 5, 6, 7, 0, 1, 2, 3, 0, 1, 2, 3],
        )

    def test_parallel(self):
        import multiprocessing
        import time

        def task(i, path, lock):
            archive = starsmashertools.lib.archive2.Archive(path)
            with lock:
                archive['test '+str(i)] = i
                archive.save()
        
        manager = multiprocessing.Manager()
        lock = manager.Lock()
        processes = [multiprocessing.Process(
            target = task,
            args = (i, Test.filename, lock),
            daemon = True,
        ) for i in range(2)]

        for process in processes: process.start()
        for process in processes: process.join()

        archive = starsmashertools.lib.archive2.Archive(Test.filename)

        self.assertEqual(2, len(self.archive))
        self.assertIn('test 0', self.archive)
        self.assertIn('test 1', self.archive)
        
        

class TestLoader(unittest.TestLoader, object):
    def getTestCaseNames(self, *args, **kwargs):
        return [
            'test_setitem',
            'test_contains',
            'test_delitem',
            'test_simple',
            'test_keys',
            'test_values',
            'test_items',
            'test_size',
            'test_readonly',
            'test_add',
            'test_complex',
            'test_parallel',
        ]

if __name__ == '__main__':
    import inspect
    import re

    comment_checks = [
        # Remove # comments first
        re.compile("(?<!['\"])#.*", flags = re.M),
        # Then remove block comments (which can be commented out by #)
        re.compile('(?<!\')(?<!\\\\)""".*?"""', flags = re.M | re.S),
        re.compile("(?<!\")(?<!\\\\)'''.*?'''", flags = re.M | re.S),
    ]

    src = inspect.getsource(starsmashertools.lib.archive2)

    # Remove all comments
    for check in comment_checks:
        for match in check.findall(src):
            src = src.replace(match, '')

    if '@profile' in src:
        loader = TestLoader()
        suite = unittest.TestSuite()
        for name in loader.getTestCaseNames():
            suite.addTest(Test(name))
        runner = unittest.TextTestRunner()
        runner.run(suite)
    else:
        # This is the normal method
        unittest.main(failfast=True, testLoader=TestLoader())

