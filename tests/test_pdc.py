import unittest
import starsmashertools.mpl.plotdata
import starsmashertools.preferences
import os
import copy
import glob

curdir = os.path.dirname(__file__)

class TestPDCFile(unittest.TestCase):
    def setUp(self):
        pdcs = glob.glob(os.path.join(curdir, "*.pdc.gz"))
        for pdc in pdcs: os.remove(pdc)
    
    def tearDown(self):
        pdcs = glob.glob(os.path.join(curdir, "*.pdc.gz"))
        for pdc in pdcs: os.remove(pdc)
    
    def test_init(self):
        pdcfile = starsmashertools.mpl.plotdata.PDCFile()
        dirname = starsmashertools.helpers.path.dirname(pdcfile.filename)
        curdir = os.path.dirname(__file__)
        
        self.assertEqual(dirname, curdir)
        self.assertEqual(len(list(pdcfile.keys())), 0)
        self.assertEqual(len(list(pdcfile.values())), 0)
        self.assertFalse(pdcfile.loaded)
        self.assertFalse(pdcfile.stale)

        pdcfile = starsmashertools.mpl.plotdata.PDCFile(filename='test')
        dirname = starsmashertools.helpers.path.dirname(pdcfile.filename)
        
        filename = os.path.join(curdir, 'test')
        
        self.assertEqual(dirname, curdir)
        self.assertEqual(pdcfile.filename, filename)
        self.assertEqual(len(list(pdcfile.keys())), 0)
        self.assertEqual(len(list(pdcfile.values())), 0)
        self.assertFalse(pdcfile.loaded)
        self.assertFalse(pdcfile.stale)

    def test_simplify_filename(self):
        pdcfile = starsmashertools.mpl.plotdata.PDCFile()
        filename = pdcfile._simplify_filename('/this/is/a/fake/test')
        self.assertEqual(filename, 'fake/test')

    def test_getitem(self):
        pdcfile = starsmashertools.mpl.plotdata.PDCFile()
        data = starsmashertools.mpl.plotdata.PDCData([], [])
        pdcfile['/this/is/a/fake/test'] = data
        self.assertIs(pdcfile['/this/is/a/fake/test'], data)
        self.assertIs(pdcfile['fake/test'], data)

    def test_setitem(self):
        pdcfile = starsmashertools.mpl.plotdata.PDCFile()
        data = starsmashertools.mpl.plotdata.PDCData([], [])
        pdcfile['/this/is/a/fake/test'] = data
        self.assertIs(pdcfile['/this/is/a/fake/test'], data)
        self.assertEqual(len(list(pdcfile.keys())), 1)
        self.assertTrue(pdcfile.stale)
        self.assertIn('fake/test', pdcfile._simplified_filenames.keys())
        self.assertEqual(pdcfile._simplified_filenames['fake/test'], '/this/is/a/fake/test')
        with self.assertRaises(ValueError):
            pdcfile['test'] = None

        class CustomType(object): pass
        
        with self.assertRaises(ValueError):
            pdcfile['fake/test'] = CustomType()

    def test_pop(self):
        pdcfile = starsmashertools.mpl.plotdata.PDCFile()
        data = starsmashertools.mpl.plotdata.PDCData([], [])
        pdcfile['/this/is/a/fake/test'] = data
        with self.assertRaises(KeyError):
            pdcfile.pop('blah')
        pdcfile.pop('/this/is/a/fake/test')
        self.assertTrue(pdcfile.stale)
        with self.assertRaises(KeyError):
            print(pdcfile._simplified_filenames['fake/test'])
        with self.assertRaises(KeyError):
            print(pdcfile['/this/is/a/fake/test'])
        
    def test_get_matching_filename(self):
        pdcfile = starsmashertools.mpl.plotdata.PDCFile()
        for i in range(5):
            pdcfile['/this/is/a/fake/test%d' % i] = starsmashertools.mpl.plotdata.PDCData([], [])

        for i in range(5):
            self.assertEqual(pdcfile._get_matching_filename('/this/is/a/fake/test%d' % i), '/this/is/a/fake/test%d' % i)
            self.assertEqual(pdcfile._get_matching_filename('fake/test%d' % i), '/this/is/a/fake/test%d' % i)

    def test_contains(self):
        pdcfile = starsmashertools.mpl.plotdata.PDCFile()
        data = starsmashertools.mpl.plotdata.PDCData([], [])
        pdcfile['/this/is/a/fake/test'] = data
        self.assertTrue(pdcfile.contains('/this/is/a/fake/test'))
        self.assertTrue(pdcfile.contains('fake/test'))

    def test_save_and_load(self):
        pdcfile = starsmashertools.mpl.plotdata.PDCFile(filename='test.pdc.gz')

        methods = ['method0', 'method1']
        values = [0, 1]
        keys = [
            '/this/is/a/fake/test0',
            '/this/is/a/fake/test1',
            '/this/is/a/fake/test2',
            '/this/is/a/fake/test3',
        ]
        for key in keys:
            pdcfile[key] = starsmashertools.mpl.plotdata.PDCData(methods, values)
        
        pdcfile.save()

        self.assertTrue(os.path.isfile(pdcfile.filename))

        pdcfile = starsmashertools.mpl.plotdata.PDCFile(filename="doesntexist.pdc.gz")
        with self.assertRaises(FileNotFoundError):
            pdcfile.load()
        
        pdcfile = starsmashertools.mpl.plotdata.PDCFile(filename='test.pdc.gz')
        pdcfile.load()
        self.assertFalse(pdcfile.stale)
        self.assertTrue(pdcfile.loaded)
        for key in keys:
            self.assertTrue(pdcfile.contains(key))
            for method, eqm, value, eqv in zip(pdcfile[key].keys(), methods, pdcfile[key].values(), values):
                self.assertEqual(method, eqm)
                self.assertEqual(value, eqv)

    def test_combine(self):
        pdcfile = starsmashertools.mpl.plotdata.PDCFile(filename='orig.pdc.gz')
        other = starsmashertools.mpl.plotdata.PDCFile(filename='other.pdc.gz')

        keys = [
            '/this/is/a/fake/test0',
            '/this/is/a/fake/test1',
            '/this/is/a/fake/test2',
            '/this/is/a/fake/test3',
        ]
        methods = ['method0', 'method1']
        values = [0, 1]
        for key in keys:
            pdcfile[key] = starsmashertools.mpl.plotdata.PDCData(methods, values)

        pdcfile['/this/is/a/fake/test3']['method1'] = None
        pdcfile.save()

        other['/this/is/a/fake/test0'] = starsmashertools.mpl.plotdata.PDCData(['method0'], [None])
        other['/this/is/a/fake/test1'] = starsmashertools.mpl.plotdata.PDCData(['method0', 'method1'], [1, 0])
        other['/this/is/a/fake/test3'] = starsmashertools.mpl.plotdata.PDCData(['method0', 'method1', 'method2'], [1, 0, None])
        other['/this/is/a/faker/test'] = starsmashertools.mpl.plotdata.PDCData(['method0', 'method1', 'method2'], [1, 0, 10])

        other.save()
        
        pdcfilecopy = copy.deepcopy(pdcfile)
        pdcfilecopy.combine(other, overwrite=False)

        self.assertEqual(pdcfilecopy['/this/is/a/fake/test0']['method0'], 0)
        self.assertEqual(pdcfilecopy['/this/is/a/fake/test0']['method1'], 1)
        
        self.assertEqual(pdcfilecopy['/this/is/a/fake/test1']['method0'], 0)
        self.assertEqual(pdcfilecopy['/this/is/a/fake/test1']['method1'], 1)
        
        self.assertEqual(pdcfilecopy['/this/is/a/fake/test2']['method0'], 0)
        self.assertEqual(pdcfilecopy['/this/is/a/fake/test2']['method1'], 1)

        self.assertEqual(pdcfilecopy['/this/is/a/fake/test3']['method0'], 0)
        self.assertEqual(pdcfilecopy['/this/is/a/fake/test3']['method1'], 0)
        with self.assertRaises(KeyError):
            print(pdcfilecopy['/this/is/a/fake/test3']['method2'])

        self.assertEqual(pdcfilecopy['/this/is/a/faker/test']['method0'], 1)
        self.assertEqual(pdcfilecopy['/this/is/a/faker/test']['method1'], 0)
        self.assertEqual(pdcfilecopy['/this/is/a/faker/test']['method2'], 10)


        
        pdcfilecopy = copy.deepcopy(pdcfile)
        pdcfilecopy.combine(other, overwrite=True)

        self.assertEqual(pdcfilecopy['/this/is/a/fake/test0']['method0'], 0)
        self.assertEqual(pdcfilecopy['/this/is/a/fake/test0']['method1'], 1)
        
        self.assertEqual(pdcfilecopy['/this/is/a/fake/test1']['method0'], 1)
        self.assertEqual(pdcfilecopy['/this/is/a/fake/test1']['method1'], 0)
        
        self.assertEqual(pdcfilecopy['/this/is/a/fake/test2']['method0'], 0)
        self.assertEqual(pdcfilecopy['/this/is/a/fake/test2']['method1'], 1)

        self.assertEqual(pdcfilecopy['/this/is/a/fake/test3']['method0'], 1)
        self.assertEqual(pdcfilecopy['/this/is/a/fake/test3']['method1'], 0)
        with self.assertRaises(KeyError):
            print(pdcfilecopy['/this/is/a/fake/test3']['method2'])

        self.assertEqual(pdcfilecopy['/this/is/a/faker/test']['method0'], 1)
        self.assertEqual(pdcfilecopy['/this/is/a/faker/test']['method1'], 0)
        self.assertEqual(pdcfilecopy['/this/is/a/faker/test']['method2'], 10)

    def test_add(self):
        pdcfile = starsmashertools.mpl.plotdata.PDCFile()
        simulation = starsmashertools.get_simulation("data")
        output = starsmashertools.lib.output.Output(os.path.join("data", "out000.sph"), simulation)

        class GetTimeCalledException(Exception):
            pass
        
        def get_time_exception(output):
            raise GetTimeCalledException
        def get_time(output):
            return output['t']

        with self.assertRaises(GetTimeCalledException):
            pdcfile.add(output, [get_time_exception])

        self.assertEqual(len(list(pdcfile.keys())), 0)

        pdcfile.add(output, [get_time])

        self.assertEqual(len(list(pdcfile.keys())), 1)
        self.assertIn(output.path, pdcfile.keys())
        self.assertEqual(pdcfile[output.path]['get_time'], output['t'])

    def test_get_missing(self):
        pdcfile = starsmashertools.mpl.plotdata.PDCFile()
        pdcfile['/this/is/a/fake/test'] = starsmashertools.mpl.plotdata.PDCData(['m0', 'm1'], [0, 1])

        self.assertEqual(len(pdcfile.get_missing('/this/is/a/fake/test')), 0)
        self.assertEqual(len(pdcfile.get_missing('/this/is/a/fake/test', methods=['m0'])), 0)
        self.assertEqual(len(pdcfile.get_missing('/this/is/a/fake/test', methods=['m0','m1'])), 0)
        self.assertEqual(len(pdcfile.get_missing('/this/is/a/fake/test', methods=['m2'])), 1)
        self.assertEqual(len(pdcfile.get_missing('/this/is/a/fake/test', methods=['m0', 'm1', 'm2'])), 1)
        self.assertEqual(len(pdcfile.get_missing('/this/is/a/fake/testqedqw')), 1)
        self.assertEqual(len(pdcfile.get_missing('/this/is/a/fake/testqedqw', methods=['m0'])), 1)
        self.assertEqual(len(pdcfile.get_missing('/this/is/a/fake/testqedqw', methods=['m0', 'm1', 'm2'])), 1)

class TestPDCData(unittest.TestCase):
    def test_init(self):
        data = starsmashertools.mpl.plotdata.PDCData([], [])
        self.assertEqual(len(list(data.keys())), 0)
        self.assertEqual(len(list(data.values())), 0)
        data = starsmashertools.mpl.plotdata.PDCData(['method0','method1'], [0, 1])
        self.assertEqual(len(list(data.keys())), 2)
        self.assertEqual(len(list(data.values())), 2)

    def test_setitem(self):
        data = starsmashertools.mpl.plotdata.PDCData([], [])
        with self.assertRaises(KeyError):
            data[None] = []

        class CustomType(object): pass
        
        with self.assertRaises(ValueError):
            data['method'] = CustomType()

        # I think these are all the permitted JSON types
        permitted_types = ["", 0, 1.0, False, None]
        for _type in permitted_types:
            data[type(_type).__name__] = _type
        
        self.assertEqual(len(list(data.keys())), len(permitted_types))
        self.assertEqual(len(list(data.values())), len(permitted_types))

    def test_combine(self):
        data = starsmashertools.mpl.plotdata.PDCData(['m0', 'm1'], [0, None])
        other = starsmashertools.mpl.plotdata.PDCData(['m0', 'm1'], [1, 2])

        datacopy = copy.copy(data)
        datacopy.combine(other, overwrite=False)
        self.assertEqual(datacopy['m0'], 0)
        self.assertEqual(datacopy['m1'], 2)

        datacopy = copy.copy(data)
        datacopy.combine(other, overwrite=True)
        self.assertEqual(datacopy['m0'], 1)
        self.assertEqual(datacopy['m1'], 2)
        
if __name__ == '__main__':
    unittest.main(failfast=True)
