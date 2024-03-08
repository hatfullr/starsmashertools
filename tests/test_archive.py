import unittest
import starsmashertools.lib.archive
import starsmashertools.helpers.argumentenforcer
import os
import basetest

class TestArchive(basetest.BaseTest):
    def setUp(self):
        #print("setUp")
        self.archive = starsmashertools.lib.archive.Archive(
            'test.dat',
            auto_save = False,
            readonly = False,
        )

    def tearDown(self):
        self.archive.clear()
        filenames = [
            self.archive.filename,
            'test_autoSave.dat',
            'test_combine.dat',
        ]
        for filename in filenames:
            if os.path.exists(filename):
                os.remove(filename)

    def test_buffers(self):
        self.archive._buffers['add']['blah'] = None
        self.archive._buffers['remove'] += ['blah2']

        self.assertIn('blah', self.archive._buffers['add'].keys())
        self.assertIn('blah2', self.archive._buffers['remove'])

        self.archive._clear_buffers()

        self.assertNotIn('blah', self.archive._buffers['add'].keys())
        self.assertNotIn('blah2', self.archive._buffers['remove'])

        self.assertEqual(0, len(self.archive._buffers['add'].keys()))
        self.assertEqual(0, len(self.archive._buffers['remove']))

    def test_setitem(self):
        value = starsmashertools.lib.archive.ArchiveValue(
            'test',
            0,
        )
        self.archive['test'] = value
        
        self.assertIn('test', self.archive._buffers['add'].keys())
        self.assertEqual(value, self.archive._buffers['add']['test'])
        self.assertEqual(0, len(self.archive._buffers['remove']))
        self.assertEqual(1, len(self.archive._buffers['add'].keys()))

    def test_contains(self):
        value = starsmashertools.lib.archive.ArchiveValue(
            'test',
            0,
        )
        self.assertNotIn('test', self.archive)
        self.assertNotIn(value, self.archive)
        
        self.archive['test'] = value
        
        self.assertIn('test', self.archive)
        self.assertIn(value, self.archive)

        self.archive._clear_buffers()

        self.assertNotIn('test', self.archive)
        self.assertNotIn(value, self.archive)

    def test_save(self):
        value = starsmashertools.lib.archive.ArchiveValue(
            'test',
            0,
        )
        self.archive['test'] = value

        self.assertFalse(os.path.exists(self.archive.filename))
        self.archive.save()
        self.assertTrue(os.path.exists(self.archive.filename))
        
        self.assertLess(
            0,
            os.path.getsize(self.archive.filename),
        )
        
        self.assertEqual(0, len(self.archive._buffers['add'].keys()))
        self.assertEqual(0, len(self.archive._buffers['remove']))

        self.assertIn('test', self.archive)
        self.assertIn(value, self.archive)

        

    def test_keys(self):
        self.archive._buffers['add']['test'] = None
        self.assertIn('test', self.archive.keys())

        del self.archive._buffers['add']['test']

        self.assertNotIn('test', self.archive.keys())

    def test_values(self):
        value = starsmashertools.lib.archive.ArchiveValue(
            'test',
            0,
        )
        self.archive._buffers['add']['test'] = value

        self.assertEqual(value, self.archive.values()[0])

    def test_items(self):
        value = starsmashertools.lib.archive.ArchiveValue(
            'test',
            0,
        )
        self.archive._buffers['add']['test'] = value
        
        d = {key:val for key, val in self.archive.items()}
        
        self.assertEqual(1, len(d.keys()))
        self.assertEqual(1, len(d.values()))
        self.assertEqual('test', list(d.keys())[0])
        self.assertEqual(value, d['test'])
        
        self.archive.save()
        
        d = {key:val for key, val in self.archive.items()}
        
        self.assertEqual(1, len(d.keys()))
        self.assertEqual(1, len(d.values()))
        self.assertEqual('test', list(d.keys())[0])
        self.assertEqual(value, d['test'])

    def test_delitem(self):
        value = starsmashertools.lib.archive.ArchiveValue(
            'test',
            0,
        )
        self.archive['test'] = value

        del self.archive['test']

        self.assertNotIn('test', self.archive.keys())
        self.assertNotIn(value, self.archive.values())

        self.archive['test'] = value

        self.assertEqual(0, len(self.archive._buffers['remove']))
        self.assertEqual(1, len(self.archive._buffers['add'].keys()))

        self.archive.save()

        del self.archive['test']

        self.assertNotIn('test', self.archive.keys())
        self.assertNotIn(value, self.archive.values())
        
        self.archive.save()

        self.assertFalse(os.path.exists(self.archive.filename))

        self.assertNotIn('test', self.archive.keys())
        self.assertNotIn(value, self.archive.values())



        self.archive['test1'] = value
        self.archive['test2'] = value

        self.archive.save()
        size = os.path.getsize(self.archive.filename)

        del self.archive['test2']

        self.archive.save()

        self.assertTrue(os.path.exists(self.archive.filename))

        self.assertLess(
            os.path.getsize(self.archive.filename),
            size,
        )

        
    
        
        
    def test_conversion(self):
        # Try converting all the historical Archives to the newest format
        paths = []
        for filename in os.listdir('archives'):
            if filename.endswith('.py'): continue
            paths += [os.path.join('archives', filename)]

        # All these should raise no error
        for path in paths:
            error = None
            try:
                new_archive = starsmashertools.lib.archive.update_archive_version(
                    path,
                    new_path = 'test_old.dat',
                )
            except Exception as e:
                error = e
            
            if os.path.exists('test_old.dat'):
                os.remove('test_old.dat')

            if error is not None: raise(error)

    def test_add(self):
        #print("testAdd")
        with self.assertRaises(Exception):
            self.archive['test'] = 0
        self.archive.add('test', 0, 'filename', mtime = 5)
        self.assertIn('test', self.archive)
        self.assertEqual(0, self.archive['test'].value)
        self.assertEqual(5, self.archive['test'].mtime)
        self.assertEqual('filename', self.archive['test'].origin)

    def test_remove(self):
        #print("testRemove")
        self.archive.add('test', 0, 'filename', mtime = 5)
        self.archive.remove('test')
        self.assertNotIn('test', self.archive)

    def test_save(self):
        #print("testSave")
        self.archive.add('test', 0, 'filename', mtime = 5)
        
        self.assertIn('test', self.archive)
        self.assertEqual(0, self.archive['test'].value)
        self.assertEqual('filename', self.archive['test'].origin)
        self.assertEqual(5, self.archive['test'].mtime)

        self.assertFalse(os.path.exists(self.archive.filename))
        self.archive.save()
        self.assertTrue(os.path.exists(self.archive.filename))
        
        self.assertIn('test', self.archive)
        self.assertEqual(0, self.archive['test'].value)
        self.assertEqual('filename', self.archive['test'].origin)
        self.assertEqual(5, self.archive['test'].mtime)

    def test_noSaveContext(self):
        self.assertFalse(os.path.exists(self.archive.filename))
        with self.archive.nosave(): # Disable auto save temporarily
            self.archive.add('test', 0, 'filename', mtime = 5)
            self.assertIn('test', self.archive)
            self.assertEqual(0, self.archive['test'].value)
            self.assertEqual('filename', self.archive['test'].origin)
            self.assertEqual(5, self.archive['test'].mtime)
            
        self.assertFalse(os.path.exists(self.archive.filename))
        self.assertNotIn('test', self.archive)
        

    def test_autoSave(self):
        import os
        self.assertFalse(os.path.exists(self.archive.filename))
        
        self.archive.auto_save = False
        
        self.archive.add('test', 0, 'filename', mtime = 5)
        self.assertFalse(os.path.exists(self.archive.filename))
        
        self.assertIn('test', self.archive)
        self.assertEqual(0, self.archive['test'].value)
        self.assertEqual('filename', self.archive['test'].origin)
        self.assertEqual(5, self.archive['test'].mtime)
        
        self.archive.remove('test')
        self.assertNotIn('test', self.archive)

        with self.assertRaises(KeyError):
            self.archive['test']
        
        self.archive.auto_save = True
        self.archive._max_buffer_size = 0

        self.assertFalse(os.path.exists(self.archive.filename))
        self.archive.add('test', 0, 'filename', mtime = 5)
        self.assertTrue(os.path.exists(self.archive.filename))
        self.assertIn('test', self.archive)
        self.assertEqual(0, self.archive['test'].value)
        self.assertEqual('filename', self.archive['test'].origin)
        self.assertEqual(5, self.archive['test'].mtime)
        
        archive = starsmashertools.lib.archive.Archive(
            'test_autoSave.dat',
            auto_save = False,
        )
        self.assertFalse(archive.auto_save)
        self.assertFalse(os.path.exists('test_autoSave.dat'))
        archive.add('test', 0, 'filename', mtime = 5)
        self.assertFalse(archive.auto_save)
        self.assertFalse(os.path.exists('test_autoSave.dat'))
        archive.save()
        self.assertFalse(archive.auto_save)
        self.assertTrue(os.path.exists('test_autoSave.dat'))

    def test_compression(self):
        #print("testCompression")
        # Make sure the saved archive is smaller than the original data
        filename = os.path.join('data', 'log0.sph')
        with open(filename, 'r') as f:
            content = f.read()
        self.archive.add(
            'log',
            content,
            filename,
        )
        self.archive.save()
        
        archive_size = os.path.getsize(self.archive.filename)
        log_size = os.path.getsize(filename)
        self.assertLess(archive_size, log_size)

    def test_combine(self):
        self.archive.add('test', 0, 'filename', mtime = 5)
        
        other = starsmashertools.lib.archive.Archive(
            'test_combine.dat',
            auto_save = False,
        )
        other.add('test', 1, 'filename', mtime = 6)
        other.add('test2', -1, 'ff', mtime=0)
        
        self.archive.auto_save = False
        self.archive.combine(other)

        self.assertIn('test', self.archive)
        self.assertIn('test2', self.archive)
        
        self.assertEqual(self.archive['test'].value, 1)
        self.assertEqual(self.archive['test'].mtime, 6)
        self.assertEqual(self.archive['test'].origin, 'filename')

        self.assertEqual(self.archive['test2'].value, -1)
        self.assertEqual(self.archive['test2'].mtime, 0)
        self.assertEqual(self.archive['test2'].origin, 'ff')

    def test_readOnly(self):
        import starsmashertools.lib.archive
        self.archive.readonly = True
        self.archive.auto_save = False

        with self.assertRaises(starsmashertools.lib.archive.Archive.ReadOnlyError):
            self.archive.save()

        self.archive.readonly = False

    #"""
    def test_parallel(self):
        import multiprocessing
        
        def func(filename):
            import starsmashertools.lib.archive
            import time
            import warnings
            
            archive = starsmashertools.lib.archive.Archive(
                filename,
            )
            alot_of_data = {}
            for key, val in enumerate(range(100000)):
                alot_of_data[key] = val
            warnings.filterwarnings(action = 'ignore')
            archive.add(
                'same key',
                alot_of_data,
                origin = None,
                mtime = None,
            )
            warnings.resetwarnings()

        nprocs = 4
        processes = [None]*nprocs
        for i in range(nprocs):
            processes[i] = multiprocessing.Process(
                target = func,
                args = (self.archive.filename,),
                daemon = True,
            )
        for process in processes: process.start()
        # Wait for the processes to finish
        for process in processes:
            process.join()
            process.terminate()
    #"""


class TestLoader(unittest.TestLoader, object):
    def getTestCaseNames(self, obj):
        return [
            # Very basic functionality
            'test_buffers',
            'test_setitem',
            'test_contains',
            'test_save',
            'test_keys',
            'test_values',
            'test_items',
            'test_delitem',
            
            # More advanced functionality
            'test_add',
            'test_remove',
            'test_readOnly',
            'test_autoSave',
            'test_noSaveContext',
            'test_combine',
            'test_conversion',
            'test_compression',
            'test_parallel',
        ]

        
    
if __name__ == "__main__":
    unittest.main(failfast=True, testLoader=TestLoader())
