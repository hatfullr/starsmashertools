import basetest
import unittest
import os
import starsmashertools
import starsmashertools.lib.report

curdir = os.getcwd()

# We have to update this whenever a change is made to the default formatting
expected = """name                           start         current            stop
/home/hat...sts/data     1.82612 min     8.86894  hr     18.4442 day
/home/hat...sts/data     1.82612 min     8.86894  hr     18.4442 day"""


class TestReport(basetest.BaseTest):
    def setUp(self):
        path = os.path.join(curdir, 'data')
        self.simulations = [
            starsmashertools.get_simulation(path),
            starsmashertools.get_simulation(path),
        ]
    def tearDown(self):
        if os.path.isfile('test.report'): os.remove('test.report')
        del self.simulations

    def testMain(self):
        report = starsmashertools.report(self.simulations)
        self.assertEqual(str(report), expected)

    def testSaveLoad(self):
        report = starsmashertools.report(self.simulations)

        report.save('test')
        self.assertTrue(os.path.isfile('test.report'))
        loaded_report = starsmashertools.lib.report.Report.load('test.report')
        self.assertEqual(report, loaded_report)
        
        

if __name__ == "__main__":
    unittest.main(failfast=True)
