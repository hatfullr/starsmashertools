import unittest
import starsmashertools.lib.archive2
import os
import struct
import pickle
import time
import io
import re
import sys
import numpy as np

class Whacky:
    def __init__(self, obj):
        self.obj = obj
    def thing(self): pass
    def hello(self):
        for i in range(200):
            i*i
    #def __reduce__(self):
    #    return (Whacky, (self.obj,))

def get_buffer_window_string(
        _buffer : bytes,
        size : int = 40,
        left_label : str | type(None) = None,
        right_label : str | type(None) = None,
):
    curpos = _buffer.tell()

    _buffer.seek(0, 2)
    buffer_size = _buffer.tell()
    _buffer.seek(curpos)

    _buffer.seek(-min(int(0.5 * size), buffer_size - (buffer_size - curpos)), 1)
    start = _buffer.tell()
    window = _buffer.read(min(size, buffer_size - 1))
    stop = _buffer.tell()
    
    _buffer.seek(start)
    left_content = _buffer.read(curpos - start)
    right_content = _buffer.read(stop - curpos)
    first_string = str(left_content + b' ' + right_content)[2:-1]
    bottom_string = ' '*len(str(left_content)[2:-1]) + '|' + ' '*len(str(right_content)[2:-1])

    left, right = bottom_string.split('|')
    
    if left_label:
        first_string = ' '+first_string
        left = ('|{:-^' + str(len(left)) + 's}').format(' '+left_label+' ')
    if right_label:
        right = ('{:-^' + str(len(right)) + 's}|').format(' '+right_label+' ')
    
    bottom_string = '|'.join([left, right])
    bottom_string += '\n' + 'pos'.join([' '*(len(left)-1), ' '*len(right)])
    
    string = first_string
    if bottom_string.strip(): string += '\n' + bottom_string
    
    _buffer.seek(curpos)
    
    return string
        

class Test(unittest.TestCase):
    filename = 'test'
    def setUp(self):
        if os.path.exists(Test.filename): os.remove(Test.filename)
        self.archive = starsmashertools.lib.archive2.Archive(
            path = Test.filename,
            readonly = False,
        )

    def tearDown(self):
        if os.path.exists(Test.filename): os.remove(Test.filename)
        if os.path.exists('testarchiveadd'): os.remove('testarchiveadd')

    def check_composition(self):
        footer, footer_size = self.archive.get_footer()
        self.archive._buffer.seek(self.archive._buffer.size() - footer_size - 1)
        
        self.assertEqual(
            pickle.STOP,
            self.archive._buffer.read(1),
            msg = "The character just before the start of the footer isn't an end marker for a pickle object:\n%s" % get_buffer_window_string(
                self.archive._buffer,
                right_label = 'footer',
                left_label = 'not footer',
            )
        )

        self.assertEqual(
            footer,
            pickle.load(self.archive._buffer),
            msg = 'Unexpected footer',
        )
        
        self.assertEqual(
            footer_size,
            starsmashertools.lib.archive2.FOOTERSTRUCT.unpack(self.archive._buffer.read())[0] + starsmashertools.lib.archive2.FOOTERSTRUCT.size,
            msg = 'Footer has the wrong size',
        )

        self.archive._buffer.seek(0)
        self.assertEqual(
            b'\x80',
            self.archive._buffer.read(1),
            msg = "The first character in the archive isn't the starting character for a pickle object",
        )

        # Search for large blocks of nul characters
        """
        self.archive._buffer.seek(0)
        content = self.archive._buffer.read()
        blocks = 0
        previous_characters = []
        block_set = False
        s = struct.Struct('<d')
        for c in content:
            c = c.to_bytes(1, sys.byteorder)
            if c != b'\x00':
                previous_characters = []
                block_set = False
                continue
            previous_characters += [c]
            if not block_set and len(previous_characters) >= s.size:
                blocks += 1
                block_set = True
        self.assertEqual(0, blocks)
        """
        
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
        self.assertEqual('simple', self.archive['test'])
        self.assertLess(previous_size, self.archive.size())
        self.check_composition()

    def test_numpy(self):
        arr = np.zeros((1000, 1000), dtype=float)
        arr[:,500] = 1
        self.archive['test'] = arr
        self.check_composition()
        self.assertLess(self.archive._buffer[:].count(b'\x00'), 1000*1000)
        self.assertTrue(np.array_equal(self.archive['test'], arr))
        self.assertTrue((self.archive['test'][:,500] == 1).all())
        
    def test_delitem(self):
        self.archive['test'] = 'hello'
        del self.archive['test']
        self.assertNotIn('test', self.archive)
        with self.assertRaises(KeyError):
            self.archive['test']
        self.assertEqual(0, self.archive._buffer.size())
        self.assertEqual(0, self.archive.size())
        self.assertEqual(0, self.archive.get_footer()[1])

        typecount = 0
        _types = [float, bool, np.float64, np.int8, np.ndarray]
        for i in range(100):
            self.archive['test' + str(i)] = _types[typecount](i)
            typecount += 1
            if typecount >= len(_types): typecount = 0
        self.assertEqual(100, len(self.archive))
        self.assertGreater(self.archive.size(), 0)
        
        del self.archive['test0']
        self.assertEqual(99, len(self.archive))
        self.check_composition()

        for i in range(99, 1, -1):
            del self.archive['test'+str(i)]
            self.assertEqual(i - 1, len(self.archive))
            self.check_composition()

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
        self.assertEqual(0, archive.size())

    def test_add(self):
        self.archive['test'] = 'hi'
        self.archive.add('test', 0)
        self.assertEqual(0, self.archive['test'])

        self.check_composition()

        time.sleep(1.e-4)
        open('testarchiveadd', 'x').close()
        self.archive.add(
            'test',
            0,
            origin = 'testarchiveadd',
            replace = 'mtime',
        )
        self.assertEqual(0, self.archive['test'])

        self.check_composition()

        self.archive['test'] = 'hi'
        self.archive.add(
            'test',
            0,
            origin = 'testarchiveadd',
            replace = 'mtime',
        )
        self.assertEqual('hi', self.archive['test'])

        self.check_composition()

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

    def test_append(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({
            'level 1' : { 'level 2' : { 'test' : 'hi' } }
        })
        self.assertEqual(0, len(self.archive))
        self.archive['test'] = d
        self.archive.append('test', d)
        self.assertEqual([d, d], list(self.archive['test']))
        self.check_composition()

        self.archive.append('test', d)
        self.assertEqual([d,d,d], list(self.archive['test']))
        self.check_composition()

        self.archive['test 2'] = d
        self.archive.append('test 2', d)
        self.assertEqual([d, d], list(self.archive['test 2']))
        self.check_composition()

        self.archive.clear()
        self.archive['test'] = 0
        self.archive.append('test', 1)
        self.assertEqual([0, 1], list(self.archive['test']))
        self.archive['test2'] = 0
        self.archive.append('test2', 55)
        self.assertEqual([0, 55], list(self.archive['test2']))

        for branch, leaf in d.flowers():
            self.archive[str(branch)] = leaf
        self.assertEqual('hi', self.archive[str(('level 1', 'level 2', 'test'))])
        self.check_composition()


        d2 = starsmashertools.helpers.nesteddict.NestedDict({
            'level 1' : { 'level 2' : { 'test' : 'hi2' } }
        })
        
        for branch, leaf in d2.flowers():
            key = str(branch)
            if key in self.archive: self.archive.append(key, leaf)
            else: self.archive[key] = leaf
        self.assertEqual(['hi','hi2'], list(self.archive[str(('level 1', 'level 2', 'test'))]))

        self.check_composition()


    def test_parallel(self):
        import multiprocessing
        import time

        def task(i, path, lock):
            archive = starsmashertools.lib.archive2.Archive(path, readonly = False)
            with lock:
                archive['test append'] = 0
                for j in range(10):
                    archive['test '+str(i + j)] = i + j
                    archive.append('test append', Whacky(float(i+j)))
                archive.save()
        
        manager = multiprocessing.Manager()
        lock = manager.Lock()
        processes = [multiprocessing.Process(
            target = task,
            args = (i, Test.filename, lock),
            daemon = True,
        ) for i in range(4)]

        for process in processes: process.start()
        for process in processes: process.join()

        archive = starsmashertools.lib.archive2.Archive(Test.filename)

        self.check_composition()

        #self.assertEqual(2, len(self.archive))
        #self.assertIn('test 0', self.archive)
        #self.assertIn('test 1', self.archive)
        
        

class TestLoader(unittest.TestLoader, object):
    def getTestCaseNames(self, *args, **kwargs):
        return [
            'test_setitem',
            'test_contains',
            'test_size',
            'test_delitem',
            'test_simple',
            'test_keys',
            'test_values',
            'test_items',
            'test_readonly',
            'test_add',
            'test_complex',
            'test_append',
            'test_numpy',
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

