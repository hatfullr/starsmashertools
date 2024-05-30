import unittest
import os
import starsmashertools.lib.flux
import basetest
import test_fluxresult
import starsmashertools.helpers.nesteddict

class TestLoader(unittest.TestLoader, object):
    def getTestCaseNames(self, *args, **kwargs):
        return [
            #'testLoad',
            #'testSave',
            #'testAdd',
        ]
    
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
            result = starsmashertools.lib.flux.FluxResults.load(
                os.path.join(test_fluxresult.fluxdir, details['filename']),
            )
            
            # Save as a test file
            result.save(TestFluxResults.test_filename)
            self.assertTrue(os.path.exists(TestFluxResults.test_filename))
            os.remove(TestFluxResults.test_filename)
    
    def testAdd(self):
        details = test_fluxresult.load_fluxresult_details().fluxresult_details
        
        total = starsmashertools.lib.flux.FluxResults()
        for i, detail in enumerate(details):
            path = os.path.join(test_fluxresult.fluxdir, detail['filename'])
            result = starsmashertools.lib.flux.FluxResult.load(path)
            total.add(result)
            for branch, leaf in total.flowers():
                self.assertEqual(i + 1, len(leaf))
        total.save(TestFluxResults.test_filename)
        
        new = starsmashertools.lib.flux.FluxResults.load(TestFluxResults.test_filename)
        for branch, leaf in new.flowers():
            self.assertIn(branch, total.branches())
            self.assertEqual(leaf, total[branch])
        for branch, leaf in total.flowers():
            self.assertIn(branch, new.branches())
            self.assertEqual(leaf, new[branch])


        # Try with a simple, non-nested "allowed" keyword
        total = starsmashertools.lib.flux.FluxResults(allowed = {'time' : True})
        for i, detail in enumerate(details):
            path = os.path.join(test_fluxresult.fluxdir, detail['filename'])
            result = starsmashertools.lib.flux.FluxResult.load(path)
            total.add(result)
            self.assertEqual(1, len(total.branches()))
            self.assertEqual('time', list(total.branches())[0])
            for branch, leaf in total.flowers():
                self.assertEqual(i + 1, len(leaf))

        # Try with a difficult, nested 'allowed' keyword
        total = starsmashertools.lib.flux.FluxResults(allowed = {
            'kwargs' : {
                'verbose' : True,
            },
        })
        for i, detail in enumerate(details):
            path = os.path.join(test_fluxresult.fluxdir, detail['filename'])
            result = starsmashertools.lib.flux.FluxResult.load(path)
            total.add(result)
            for branch, leaf in total.flowers():
                self.assertEqual(i + 1, len(leaf))
            self.assertEqual(1, len(total.branches()))

        # Try with the most difficult nested 'allowed' keyword
        total = starsmashertools.lib.flux.FluxResults(allowed = {
            'simulation' : True,
            'kwargs' : True,
        })
        i = 0
        for detail in details:
            path = os.path.join(test_fluxresult.fluxdir, detail['filename'])
            result = starsmashertools.lib.flux.FluxResult.load(path)
            total.add(result)
            i += 1
            for branch, leaf in total.flowers():
                self.assertEqual(i, len(leaf))
            self.assertNotEqual(2, len(total.branches()))
            self.assertIn('simulation', total.branches())
        
        os.remove(TestFluxResults.test_filename)
        total.save(TestFluxResults.test_filename)

        # Try add after save
        for detail in details:
            path = os.path.join(test_fluxresult.fluxdir, detail['filename'])
            result = starsmashertools.lib.flux.FluxResult.load(path)
            total.add(result)
            i += 1
            for branch, leaf in total.flowers():
                self.assertEqual(i, len(leaf))

        os.remove(TestFluxResults.test_filename)
        total.save(TestFluxResults.test_filename)

        new = starsmashertools.lib.flux.FluxResults.load(TestFluxResults.test_filename)
        for branch, leaf in new.flowers():
            self.assertIn(branch, total.branches())
            self.assertEqual(leaf, total[branch])
        for branch, leaf in total.flowers():
            self.assertIn(branch, new.branches())
            self.assertEqual(leaf, new[branch])

if __name__ == '__main__':
    unittest.main(failfast = True, testLoader = TestLoader())
        

    
