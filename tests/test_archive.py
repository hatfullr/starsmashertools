import unittest
import starsmashertools.lib.archive
import starsmashertools.helpers.argumentenforcer
import os
import basetest


class TestArchive(basetest.BaseTest):
    def setUp(self):
        self.archive = starsmashertools.lib.archive.Archive('test.dat')

    def tearDown(self):
        filenames = [
            self.archive.filename,
            'test_autoSave.dat',
            'test_combine.dat',
            #'test_password.dat',
        ]
        for filename in filenames:
            if os.path.isfile(filename):
                os.remove(filename)

    def testAdd(self):
        self.archive.add('test', 0, 'filename', mtime = 5)
        error = starsmashertools.helpers.argumentenforcer.ArgumentTypeError
        with self.assertRaises(Exception):
            self.archive['test'] = 0
        self.assertIn('test', self.archive)
        self.assertEqual(self.archive['test'].value, 0)
        self.assertEqual(self.archive['test'].mtime, 5)
        self.assertEqual(self.archive['test'].origin, 'filename')

    def testRemove(self):
        self.archive.add('test', 0, 'filename', mtime = 5)
        self.archive.remove('test')
        self.assertNotIn('test', self.archive)

    def testSave(self):
        self.archive.save()
        self.assertTrue(os.path.isfile(self.archive.filename))

    def testAutoSave(self):
        self.archive.auto_save = False
        self.archive.add('test', 0, 'filename', mtime = 5)
        with self.assertRaises(FileNotFoundError):
            self.archive.load()
        self.archive.remove('test')

        self.assertNotIn('test', self.archive)
        
        self.archive.auto_save = True
        self.archive.add('test', 0, 'filename', mtime = 5)
        self.archive.load()
        self.assertIn('test', self.archive)


        archive = starsmashertools.lib.archive.Archive(
            'test_autoSave.dat',
            load = False,
            auto_save = False,
        )
        self.assertFalse(archive.auto_save)
        self.assertFalse(os.path.exists('test_autoSave.dat'))
        with self.assertRaises(FileNotFoundError):
            archive.load()
        self.assertFalse(archive.auto_save)
        self.assertFalse(os.path.exists('test_autoSave.dat'))
        archive.add('test', 0, 'filename', mtime = 5)
        self.assertFalse(archive.auto_save)
        self.assertFalse(os.path.exists('test_autoSave.dat'))
        archive.save()
        self.assertFalse(archive.auto_save)
        self.assertTrue(os.path.exists('test_autoSave.dat'))
        

    def testLoad(self):
        self.archive.add('test', 0, 'filename', mtime = 5)

        new_archive = starsmashertools.lib.archive.Archive(
            self.archive.filename,
            load = True,
        )
        self.assertIn('test', new_archive)
        self.assertEqual(new_archive['test'].value, 0)
        self.assertEqual(new_archive['test'].mtime, 5)
        self.assertEqual(new_archive['test'].origin, 'filename')

        new_archive = starsmashertools.lib.archive.Archive(
            self.archive.filename,
            load = False,
        )
        new_archive.load()
        self.assertIn('test', new_archive)
        self.assertEqual(new_archive['test'].value, 0)
        self.assertEqual(new_archive['test'].mtime, 5)
        self.assertEqual(new_archive['test'].origin, 'filename')


    def testCompression(self):
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
        self.archive.add('test', 0, 'filename', mtime = 5)
        
        other = starsmashertools.lib.archive.Archive(
            'test_combine.dat', load=False,
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
        

    def testPassword(self):
        import time
        self.archive = starsmashertools.lib.archive.Archive(
            'test_password.dat',
            password='1234',
        )
        self.archive.add('test', 0, 'filename', mtime = 5)
        self.archive.save()
        self.assertTrue(os.path.isfile(self.archive.filename))
        print(self.archive.filename)
        time.sleep(100)

        #archive = starsmashertools.lib.archive.Archive(self.archive.filename)

if __name__ == "__main__":
    unittest.main(failfast=True)
