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


    def testMask(self):
        import copy
        output = self.simulation.get_output(0)
        ntot = output['ntot']
        mask = np.full(ntot, False, dtype=bool)
        idx = int(0.5*ntot)
        mask[:idx] = True

        orig_output = copy.deepcopy(output)
        output.mask(mask)

        for key, val in orig_output.items():
            if isinstance(val, np.ndarray):
                self.assertTrue(np.array_equal(val[:idx], output[key]))
            else:
                self.assertAlmostEqual(val, output[key])

        output.unmask()

        for key, val in orig_output.items():
            if isinstance(val, np.ndarray):
                self.assertTrue(np.array_equal(val, output[key]))
            else:
                self.assertAlmostEqual(val, output[key])

    def test_condense(self):
        import numpy as np
        import time

        # Make sure we don't save anything
        with self.simulation.archive.nosave():
            output = self.simulation.get_output(0)
            relpath = output._get_relpath()
            
            output.condense(
                'Eint',
                func = """import numpy as np
result = np.sum(am*u)
""",
            )

            self.assertIn('Output.condense', self.simulation.archive.keys())
            v = self.simulation.archive['Output.condense'].value[relpath]
            self.assertAlmostEqual(
                np.sum(output['am']*output['u']),
                v['Eint']['value'],
            )
            self.assertEqual(
                v['Eint']['value'],
                output.condense('Eint'),
            )

            with self.assertRaises(KeyError):
                output.condense('nonsense')

            with self.assertRaises(NameError):
                output.condense('blahblah', func = 'garbage')

            with self.assertRaises(SyntaxError):
                output.condense(
                    'Eint',
                    func = """import numpy as np
np.sum(am*u)
""",
                    overwrite = True,
                )

            current_time = time.time()
            v = self.simulation.archive['Output.condense'].value[relpath]
            self.assertGreater(
                current_time,
                v['Eint']['access time'],
            )

            output.condense('Blah', func = "result = 1+1")
            v = self.simulation.archive['Output.condense'].value[relpath]
            self.assertGreater(
                v['Blah']['access time'],
                v['Eint']['access time'],
            )

            def func(o):
                import numpy as np
                return np.sum(o['am']*o['u']) + 1
            v = self.simulation.archive['Output.condense'].value[relpath]
            old = v['Eint']['value']
            output.condense('Eint', func = func, overwrite = True)
            v = self.simulation.archive['Output.condense'].value[relpath]
            self.assertGreater(
                v['Eint']['value'],
                old,
            )

if __name__ == "__main__":
    unittest.main(failfast=True)
