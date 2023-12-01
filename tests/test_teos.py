import unittest
import os
import starsmashertools
import starsmashertools.lib.teos

class TestTEOS(unittest.TestCase):
    def setUp(self):
        simdir = os.path.join(
            starsmashertools.SOURCE_DIRECTORY,
            'tests',
            'data',
        )
        self.simulation = starsmashertools.get_simulation(simdir)

    def testSimulationHasTEOS(self):
        self.assertNotEqual(self.simulation.teos, None)
        self.assertTrue(isinstance(self.simulation.teos, starsmashertools.lib.teos.TEOS))

    def testKeys(self):
        keys = self.simulation.teos.keys()
        expected = [
            'log Rho',
            'log U',
            'log T',
            'mu',
            'P',
            'S_gas?',
            'dlnP/dlnT',
            'dlnU/dlnT',
        ]
        for key in expected:
            self.assertIn(key, keys)
        for key in keys:
            self.assertIn(key, expected)

    def testInfo(self):
        self.assertEqual(0.70000, self.simulation.teos.info['X'])
        self.assertEqual(0.28000, self.simulation.teos.info['Y'])
        self.assertEqual(0.02000, self.simulation.teos.info['Z'])
        self.assertEqual(1.29664, self.simulation.teos.info['abar'])
        self.assertEqual(1.10214, self.simulation.teos.info['zbar'])

        self.assertEqual(5, self.simulation.teos.info['log rho num'])
        self.assertEqual(459, self.simulation.teos.info['log U num'])
        self.assertEqual(-14.0000, self.simulation.teos.info['log rho min'])
        self.assertEqual(-13.9000, self.simulation.teos.info['log rho max'])
        self.assertEqual(0.0250, self.simulation.teos.info['log rho step'])
        self.assertEqual(9.8400, self.simulation.teos.info['log U min'])
        self.assertEqual(19.0000, self.simulation.teos.info['log U max'])
        self.assertEqual(0.0200, self.simulation.teos.info['log U step'])
        
    def testBodyInterpolation(self):
        # Testing if interpolation is working properly in the body of the table
        x1, x2 = -13.95, -13.925
        y1, y2 = 9.84, 9.86
        Q11, Q12, Q22, Q21 = 2.1079, 2.1263, 2.1270, 2.1085

        # -13.9500     9.8400     2.1079
        # -13.9500     9.8600     2.1263
        # -13.9250     9.8400     2.1085
        # -13.9250     9.8600     2.1270

        x = 0.5 * (x1 + x2)
        y = 0.5 * (y1 + y2)

        expected = (Q11 + Q12 + Q22 + Q21) * 0.25
        found = self.simulation.teos(10**x, 10**y, 'log T')
        self.assertAlmostEqual(expected, found)

        x = x1 + 0.1 * (x2 - x1)
        y = y1 + 0.1 * (y2 - y1)

        numerator = Q11*(x2-x)*(y2-y) + Q21*(x-x1)*(y2-y) + Q12*(x2-x)*(y-y1) + Q22*(x-x1)*(y-y1)
        denominator = (x2-x1)*(y2-y1)

        expected = numerator / denominator
        found = self.simulation.teos(10**x, 10**y, 'log T')
        self.assertAlmostEqual(expected, found)

        for x, y, expected in [[x1, y1, Q11], [x1, y2, Q12], [x2, y1, Q21], [x2, y2, Q22]]:
            found = self.simulation.teos(10**x, 10**y, 'log T')
            self.assertAlmostEqual(expected, found)


    def testUpperBoundaryInterpolation(self):
        #   -14.0000     9.8400     2.1068
        #   -14.0000     9.8600     2.1251
        #   -13.9750     9.8400     2.1072
        #   -13.9750     9.8600     2.1256
        x1, x2 =  -14, -13.975
        y1, y2 = 9.84,    9.86
        Q11, Q12, Q21, Q22 = 2.1068, 2.1251, 2.1072, 2.1256

        x = 0.5 * (x1 + x2)
        y = 0.5 * (y1 + y2)
        
        expected = (Q11 + Q12 + Q21 + Q22) * 0.25
        found = self.simulation.teos(10**x, 10**y, 'log T')
        self.assertAlmostEqual(expected, found)

        for x, y, expected in [[x1, y1, Q11], [x1, y2, Q12], [x2, y1, Q21], [x2, y2, Q22]]:
            found = self.simulation.teos(10**x, 10**y, 'log T')
            self.assertAlmostEqual(expected, found)

        oob = [
            [x1-0.01,      y1],
            [x1-0.01,      y2],
            [     x1, y1-0.01],
            [     x2, y1-0.01],
            [x1-0.01, y1-0.01],
        ]
        not_oob = [
            [x2+0.01,      y1],
            [x2+0.01,      y2],
            [     x1, y2+0.01],
            [     x2, y2+0.01],
            [x2+0.01, y2+0.01],
        ]
            
        for x, y in oob:
            with self.assertRaises(
                    starsmashertools.lib.teos.Interpolator.OutOfBoundsError):
                self.simulation.teos(10**x, 10**y, 'log T')
        for x, y in not_oob:
            self.simulation.teos(10**x, 10**y, 'log T')


    def testLowerBoundaryInterpolation(self):
        #   -13.9250    18.9800     4.7940
        #   -13.9250    19.0000     4.7990
        #   -13.9000    18.9800     4.8003
        #   -13.9000    19.0000     4.8053
        x1, x2 = -13.925, -13.9
        y1, y2 =   18.98,  19.0
        Q11, Q12, Q21, Q22 = 4.794, 4.7990, 4.8003, 4.8053

        x = 0.5 * (x1 + x2)
        y = 0.5 * (y1 + y2)
        expected = (Q11 + Q12 + Q21 + Q22) * 0.25
        found = self.simulation.teos(10**x, 10**y, 'log T')
        self.assertAlmostEqual(expected, found)

        for x, y, expected in [[x1, y1, Q11], [x1, y2, Q12], [x2, y1, Q21], [x2, y2, Q22]]:
            found = self.simulation.teos(10**x, 10**y, 'log T')
            self.assertAlmostEqual(expected, found)

        not_oob = [
            [x1-0.01,      y1],
            [x1-0.01,      y2],
            [     x1, y1-0.01],
            [     x2, y1-0.01],
            [x1-0.01, y1-0.01],
        ]
        oob = [
            [x2+0.01,      y1],
            [x2+0.01,      y2],
            [     x1, y2+0.01],
            [     x2, y2+0.01],
            [x2+0.01, y2+0.01],
        ]

        for x, y in oob:
            with self.assertRaises(
                    starsmashertools.lib.teos.Interpolator.OutOfBoundsError):
                self.simulation.teos(10**x, 10**y, 'log T')

        for x, y in not_oob:
            self.simulation.teos(10**x, 10**y, 'log T')

    def testMultipleValues(self):
        import numpy as np

        xyz = np.array([
            [-14, 9.84, 2.1068],
            [-14, 9.85, 0.5*(2.1251 + 2.1068)],
            [-13.9, 18.98, 4.8003],
            [-13.9, 19.0, 4.8053],
            [-13.9, 18.99, 0.5*(4.8003 + 4.8053)],
        ])
        
        result = self.simulation.teos(10**xyz[:,0], 10**xyz[:,1], 'log T')
        for i, (found, expected) in enumerate(zip(result, xyz[:,2])):
            self.assertAlmostEqual(expected, found, msg="x = %f, y = %f" % (xyz[i][0], xyz[i][1]))
        

if __name__ == "__main__":
    unittest.main(failfast=True)
