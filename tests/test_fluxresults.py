import unittest
import os
import starsmashertools.lib.flux
import basetest
import test_fluxresult
import starsmashertools.helpers.nesteddict

class TestLoader(unittest.TestLoader, object):
    def getTestCaseNames(self, *args, **kwargs):
        return [
            'testSave',
            'testAdd',
        ]
    
class TestFluxResults(basetest.BaseTest):

    test_filename = 'fluxresults_test.dat'
    
    def setUp(self):
        # Load in the details about how the FluxResult objects were made
        self.fluxresult_details = test_fluxresult.load_fluxresult_details().fluxresults_details

    def tearDown(self):
        if os.path.exists(TestFluxResults.test_filename):
            os.remove(TestFluxResults.test_filename)

    def _check_values(self, obj):
        import types
        if isinstance(obj.values(), types.GeneratorType):
            expected = list(obj.values())[0]
        else:
            expected = obj.values()[0]
        if isinstance(expected, types.GeneratorType):
            expected = len(list(expected))
        else: expected = 1
        for val in obj.values():
            if isinstance(val, types.GeneratorType): val = len(list(val))
            else: val = 1
            self.assertEqual(expected, val)

    
    def testSave(self):
        import gzip
        import mmap
        import pickle
        for details in self.fluxresult_details:
            result = starsmashertools.lib.flux.FluxResults.load(
                os.path.join(test_fluxresult.fluxdir, details['filename']),
            )
            
            # Save as a test file
            result.save(TestFluxResults.test_filename)
            self.assertTrue(os.path.exists(TestFluxResults.test_filename))

            with gzip.open(TestFluxResults.test_filename, 'rb') as f:
                print(pickle.load(f))
                _buffer = mmap.mmap(f.fileno(), 0, access = mmap.ACCESS_READ)
                pos = _buffer.rfind(pickle.STOP, 0)
                while pos != -1:
                    _buffer.seek(pos)
                    
                    try: print(pickle.load(_buffer))
                    except: pass
                    pos = _buffer.rfind(pickle.STOP, 0, pos - 1)
                    print(pos)
                #print(_buffer[_buffer.tell():])
                #print(pickle.load(_buffer))
                
            quit()
            os.remove(TestFluxResults.test_filename)
    
    def testAdd(self):
        details = test_fluxresult.load_fluxresult_details().fluxresult_details
        
        total = starsmashertools.lib.flux.FluxResults()
        for i, detail in enumerate(details):
            path = os.path.join(test_fluxresult.fluxdir, detail['filename'])
            result = starsmashertools.lib.flux.FluxResult.load(path)
            total.add(result)
            for branch, leaf in total.flowers():
                self.assertEqual(i + 1, len(leaf), msg=str(branch))
        total.save(TestFluxResults.test_filename)
        

if __name__ == '__main__':
    unittest.main(failfast = True, testLoader = TestLoader())
        

    
