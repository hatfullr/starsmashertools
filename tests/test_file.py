import starsmashertools.helpers.file
import unittest
import os

curdir = os.getcwd()

class TestFile(unittest.TestCase):
    def test_get_phrase(self):
        result = starsmashertools.helpers.file.get_phrase(
            os.path.join('data', 'testfile'),
            'test',
        )
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], '4')
        self.assertEqual(result[1], '9')
        self.assertEqual(result[2], '')
        self.assertEqual(result[3], '6789')

        result = starsmashertools.helpers.file.get_phrase(
            os.path.join('data', 'testfile1'),
            'test',
        )
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], '4')
        self.assertEqual(result[1], '901234M6789012345')
        self.assertEqual(result[2], '01')
        self.assertEqual(result[3], '6789')


if __name__ == "__main__":
    unittest.main()

