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

        with self.assertWarns(UserWarning):
            new_archive.add('test', 1, 'filename', mtime = 4)
        self.assertTrue(new_archive['test'].value == 0)
        self.assertTrue(new_archive['test'].mtime == 5)
        self.assertTrue(new_archive['test'].origin == 'filename')

        new_archive.add('test', 1, 'filename', mtime = 6)
        self.assertTrue(new_archive['test'].value == 1)
        self.assertTrue(new_archive['test'].mtime == 6)
        self.assertTrue(new_archive['test'].origin == 'filename')
        

if __name__ == "__main__":
    unittest.main(failfast=True)
