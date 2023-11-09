import unittest
import starsmashertools.lib.archive
import os


class TestArchive(unittest.TestCase):
    def setUp(self):
        self.archive = starsmashertools.lib.archive.Archive('test.dat')

    def tearDown(self):
        if os.path.isfile(self.archive.filename):
            os.remove(self.archive.filename)
    
    def test(self):
        self.archive.add('test', 0, 'filename', mtime = 5)
        with self.assertRaises(Exception):
            self.archive['test'] = 0

        self.assertTrue('test' in self.archive.keys())
        self.assertTrue(self.archive['test'].value == 0)
        self.assertTrue(self.archive['test'].mtime == 5)
        self.assertTrue(self.archive['test'].origin == 'filename')

        self.archive.save()
        self.assertTrue(os.path.isfile(self.archive.filename))
        
        new_archive = starsmashertools.lib.archive.Archive(self.archive.filename)
        self.assertTrue('test' in new_archive.keys())
        self.assertTrue(new_archive['test'].value == 0)
        self.assertTrue(new_archive['test'].mtime == 5)
        self.assertTrue(new_archive['test'].origin == 'filename')

        #with self.assertWarns(UserWarning):
        #    new_archive.add('test', 1, 'filename', mtime = 4)
        self.assertTrue(new_archive['test'].value == 0)
        self.assertTrue(new_archive['test'].mtime == 5)
        self.assertTrue(new_archive['test'].origin == 'filename')

        new_archive.add('test', 1, 'filename', mtime = 6)
        self.assertTrue(new_archive['test'].value == 1)
        self.assertTrue(new_archive['test'].mtime == 6)
        self.assertTrue(new_archive['test'].origin == 'filename')

        new_archive.remove('test')
        self.assertFalse('test' in new_archive.keys())

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
        
        new_archive = starsmashertools.lib.archive.Archive(self.archive.filename)
        self.assertEqual(new_archive['log'].value, content)

    def test_combine(self):
        self.archive.add('test', 0, 'filename', mtime = 5)
        
        other = starsmashertools.lib.archive.Archive('test2.dat')
        other.add('test', 1, 'filename', mtime = 6)
        other.add('test2', -1, 'ff', mtime=0)
        
        self.archive.combine(other, save = False)
        self.assertTrue(self.archive['test'].value == 1)
        self.assertTrue(self.archive['test'].mtime == 6)
        self.assertTrue(self.archive['test'].origin == 'filename')

        self.assertTrue(self.archive['test2'].value == -1)
        self.assertTrue(self.archive['test2'].mtime == 0)
        self.assertTrue(self.archive['test2'].origin == 'ff')
        

if __name__ == "__main__":
    unittest.main(failfast=True)
