import basetest
import unittest
import os
import starsmashertools

curdir = os.getcwd()

# We have to update this whenever a change is made to the default formatting
expected = """                name        start         end
...rtools/tests/data  1.82612 min  8.86894 hr
...rtools/tests/data  1.82612 min  8.86894 hr"""


class TestReport(basetest.BaseTest):
    def setUp(self):
        path = os.path.join(curdir, 'data')
        self.simulations = [
            starsmashertools.get_simulation(path),
            starsmashertools.get_simulation(path),
        ]
    def tearDown(self):
        if os.path.isfile('test_report'): os.remove('test_report')
        del self.simulations

    def testMain(self):
        report = starsmashertools.report(self.simulations)
        string = report.write()
        self.assertEqual(string, expected)
        with open('test_report', 'w') as f:
            report.write(f)
        self.assertTrue(os.path.isfile('test_report'))
        with open('test_report', 'r') as f:
            self.assertEqual(f.read(), expected)

if __name__ == "__main__":
    unittest.main(failfast=True)
