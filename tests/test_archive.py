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
            auto_save = True,
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
            if os.path.isfile(filename):
                os.remove(filename)

    def testConversion(self):
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
            
            if os.path.isfile('test_old.dat'):
                os.remove('test_old.dat')

            if error is not None: raise(error)

    def testAdd(self):
        #print("testAdd")
        with self.assertRaises(Exception):
            self.archive['test'] = 0
        self.archive.add('test', 0, 'filename', mtime = 5)
        self.assertIn('test', self.archive)
        self.assertEqual(0, self.archive['test'].value)
        self.assertEqual(5, self.archive['test'].mtime)
        self.assertEqual('filename', self.archive['test'].origin)

    def testRemove(self):
        #print("testRemove")
        self.archive.add('test', 0, 'filename', mtime = 5)
        self.archive.remove('test')
        self.assertNotIn('test', self.archive)

    def testSave(self):
        #print("testSave")
        with self.archive.nosave(): # Disable auto save temporarily
            self.archive.add('test', 0, 'filename', mtime = 5)

        self.assertIn('test', self.archive)
        self.assertEqual(1, len(list(self.archive._to_add.keys())))
        self.assertEqual(0, len(self.archive._to_remove))
        self.assertEqual(0, self.archive['test'].value)
        self.assertEqual('filename', self.archive['test'].origin)
        self.assertEqual(5, self.archive['test'].mtime)

        self.assertFalse(os.path.isfile(self.archive.filename))
        self.archive.save()
        self.assertTrue(os.path.isfile(self.archive.filename))
        
        self.assertIn('test', self.archive)
        self.assertEqual(0, len(list(self.archive._to_add.keys())))
        self.assertEqual(0, len(self.archive._to_remove))
        self.assertEqual(0, self.archive['test'].value)
        self.assertEqual('filename', self.archive['test'].origin)
        self.assertEqual(5, self.archive['test'].mtime)

    def testNoSaveContext(self):
        self.assertFalse(os.path.isfile(self.archive.filename))
        with self.archive.nosave(): # Disable auto save temporarily
            self.archive.add('test', 0, 'filename', mtime = 5)
        
        self.assertFalse(os.path.isfile(self.archive.filename))
        self.assertIn('test', self.archive)
        self.assertEqual(1, len(list(self.archive._to_add.keys())))
        self.assertEqual(0, len(self.archive._to_remove))
        self.assertEqual(0, self.archive['test'].value)
        self.assertEqual('filename', self.archive['test'].origin)
        self.assertEqual(5, self.archive['test'].mtime)

    def testAutoSave(self):
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

    def testCompression(self):
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
        
        archive_size = os.path.getsize(self.archive.filename)
        log_size = os.path.getsize(filename)
        self.assertLess(archive_size, log_size)

    def testCombine(self):
        #print("testCombine")
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

    def testReadOnly(self):
        import starsmashertools.lib.archive
        self.archive.readonly = True
        self.archive.auto_save = False

        with self.assertRaises(starsmashertools.lib.archive.Archive.ReadOnlyError):
            self.archive.save()

        self.archive.readonly = False

    #"""
    def testParallel(self):
        import multiprocessing
        
        def func(filename):
            import starsmashertools.lib.archive
            import time
            archive = starsmashertools.lib.archive.Archive(
                filename,
            )
            alot_of_data = {}
            for key, val in enumerate(range(1000000)):
                alot_of_data[key] = val
            archive.add(
                filename,
                alot_of_data,
            )

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
if __name__ == "__main__":
    unittest.main(failfast=True)
