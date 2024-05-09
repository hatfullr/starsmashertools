import unittest

badimport = False
try:
    import starsmashertools.helpers.gpujob
except RuntimeError as e:
    if 'This error is known to happen for computers' not in str(e): raise
    badimport = True
    
import starsmashertools.helpers.path
import numpy as np
import basetest

class TestGPUJob(basetest.BaseTest):
    def test(self):
        with self.assertRaises(ValueError):
            g = starsmashertools.helpers.gpujob.GPUJob([], [])

        inputs = [np.arange(0, 5)]
        outputs = [np.full(5, np.nan)]

        g = starsmashertools.helpers.gpujob.GPUJob(inputs, outputs)
        self.assertEqual(g._resolution, outputs[0].shape)
        with self.assertRaises(NotImplementedError):
            g.run()


    def testGravPot(self):
        curdir = starsmashertools.helpers.path.dirname(__file__)
        simulation = starsmashertools.get_simulation(
            starsmashertools.helpers.path.join(curdir, 'data')
        )

        output = simulation.get_output(0)
        g = starsmashertools.helpers.gpujob.GravitationalPotentialEnergies(
            output,
        )
        result = g.run()
        for r1, r2 in zip(result, output['grpot']):
            self.assertAlmostEqual(r1, r2)
        

        

if __name__ == "__main__":
    if not badimport:
        unittest.main(failfast=True)

