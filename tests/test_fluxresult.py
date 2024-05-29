import unittest
import os
import starsmashertools.lib.flux
import basetest

fluxdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'flux')

def load_fluxresult_details():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'fluxresult_details',os.path.join(fluxdir, 'fluxresult_details.py'),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class TestFluxResult(basetest.BaseTest):

    test_filename = 'fluxresult_test.zip'
    
    def setUp(self):
        # Load in the details about how the FluxResult objects were made
        self.fluxresult_details = load_fluxresult_details().fluxresult_details

    def tearDown(self):
        if os.path.exists(TestFluxResult.test_filename):
            os.remove(TestFluxResult.test_filename)
    
    def testLoad(self):
        for details in self.fluxresult_details:
            result = starsmashertools.lib.flux.FluxResult.load(
                os.path.join(fluxdir, details['filename']),
            )
            for key, val in details['allowed'].items():
                self.assertIn(key, result, msg = details['filename'])

    def testSave(self):
        for details in self.fluxresult_details:
            result = starsmashertools.lib.flux.FluxResult.load(
                os.path.join(fluxdir, details['filename']),
            )
            result.save(TestFluxResult.test_filename)
            self.assertTrue(os.path.exists(TestFluxResult.test_filename))
            os.remove(TestFluxResult.test_filename)
    

class TestLoader(unittest.TestLoader, object):
    def getTestCaseNames(self, *args, **kwargs):
        return [
            'testLoad',
            'testSave',
        ]

if __name__ == '__main__':
    unittest.main(failfast = True, testLoader = TestLoader())
        

    
