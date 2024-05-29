import unittest
import os
import starsmashertools.lib.flux
import basetest
import test_fluxresult

class TestFluxResults(basetest.BaseTest):

    test_filename = 'fluxresults_test.zip'
    
    def setUp(self):
        # Load in the details about how the FluxResult objects were made
        self.fluxresult_details = test_fluxresult.load_fluxresult_details().fluxresults_details

    def tearDown(self):
        if os.path.exists(TestFluxResults.test_filename):
            os.remove(TestFluxResults.test_filename)
    
    def testLoad(self):
        for details in self.fluxresult_details:
            result = starsmashertools.lib.flux.FluxResult.load(
                os.path.join(test_fluxresult.fluxdir, details['filename']),
            )
            # The results must be stored with only the keys which were allowed
            # to be stored.
            for key, val in details['allowed'].items():
                if not val: continue
                for k in result.keys():
                    if key in k: break
                else:
                    raise KeyError(key +' '+details['filename'])
                    
            # Each key must have a list of values of the same length as the
            # number of stored FluxResult objects.
            for branch, leaf in result.flowers():
                self.assertEqual(len(leaf), len(details['FluxResult']), msg = details['filename'])
    
    def testSave(self):
        for details in self.fluxresult_details:
            result = starsmashertools.lib.flux.FluxResult.load(
                os.path.join(test_fluxresult.fluxdir, details['filename']),
            )
            result.save(TestFluxResults.test_filename)
            self.assertTrue(os.path.exists(TestFluxResults.test_filename))
            os.remove(TestFluxResults.test_filename)
    

class TestLoader(unittest.TestLoader, object):
    def getTestCaseNames(self, *args, **kwargs):
        return [
            'testLoad',
            'testSave',
        ]

if __name__ == '__main__':
    unittest.main(failfast = True, testLoader = TestLoader())
        

    
