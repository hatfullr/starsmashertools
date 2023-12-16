import unittest
import os
import starsmashertools.lib.output
import starsmashertools
import numpy as np
import basetest

curdir = os.getcwd()

class TestOutput(basetest.BaseTest):
    expected_header = {'ntot': 1038, 'nnopt': 27, 'hco': 1.0, 'hfloor': 0.0, 'sep0': 20000.0, 'tf': 1000.0, 'dtout': 10.0, 'nout': 0, 'nit': 1, 't': 0.06875548945402724, 'nav': 3, 'alpha': 1.0, 'beta': 2.0, 'tjumpahead': 1e+30, 'ngr': 3, 'nrelax': 1, 'trelax': 1e+30, 'dt': 0.04584774736682386, 'omega2': 0.0, 'ncooling': 0, 'erad': 0.0, 'ndisplace': 0, 'displacex': 0.0, 'displacey': 0.0, 'displacez': 0.0}
    
    def setUp(self):
        path = os.path.join(curdir, 'data')
        self.simulation = starsmashertools.get_simulation(path)

    def tearDown(self):
        del self.simulation
        
    def testSingle(self):
        output = self.simulation.get_output(0)
        for key, val in output.header.items():
            self.assertAlmostEqual(val, TestOutput.expected_header[key])

    def testTraceParticle(self):
        outputs = self.simulation.get_output_iterator()
        first_output = self.simulation.get_output(0)
        particles = np.arange(len(first_output['x']))
        with starsmashertools.trace_particles(particles, self.simulation) as p:
            for particle, output in zip(p, outputs):
                for key, val in particle.items():
                    if isinstance(val, np.ndarray):
                        self.assertTrue(np.array_equal(val, output[key][particles]))
                    else:
                        self.assertTrue(val, output[key])


                    
            

if __name__ == "__main__":
    unittest.main(failfast=True)
