import unittest
import starsmashertools.helpers.extents
import starsmashertools.helpers.argumentenforcer
import numpy as np

class TestExtents(unittest.TestCase):
    def testValuedInit(self):
        extents = starsmashertools.helpers.extents.Extents(
            ymin =   np.int64(1),   ymax =           -1,
            zmin = -float('inf'),   zmax =       np.inf,
        )
        self.assertAlmostEqual(extents.xmin, -float('inf'))
        self.assertAlmostEqual(extents.xmax, float('inf'))
        self.assertAlmostEqual(extents.ymin, np.int64(1))
        self.assertAlmostEqual(extents.ymax, -1)
        self.assertAlmostEqual(extents.zmin, -float('inf'))
        self.assertAlmostEqual(extents.zmax, np.inf)

    def testGetSize(self):
        extents = starsmashertools.helpers.extents.Extents(
            xmin = 0., xmax = 1.,
            ymin = -1.e30, ymax = 1.e30,
            zmin = -np.inf, zmax = float('inf'),
        )
        self.assertAlmostEqual(extents.width, 1.)
        self.assertAlmostEqual(extents.breadth, 2.e30)
        self.assertAlmostEqual(extents.height, np.inf)
        self.assertAlmostEqual(extents.volume, np.inf)

    def testDisallowedValues(self):
        extents = starsmashertools.helpers.extents.Extents(
            xmin = -1., xmax = 1.,
            ymin = -1., ymax = 1.,
            zmin = -1., zmax = 1.,
        )
        error = starsmashertools.helpers.extents.NegativeValueError
        with self.assertRaises(error): extents.xmin = 2.
        with self.assertRaises(error): extents.xmax = -2.
        with self.assertRaises(error): extents.ymin = 2.
        with self.assertRaises(error): extents.ymax = -2.
        with self.assertRaises(error): extents.zmin = 2.
        with self.assertRaises(error): extents.zmax = -2.

    def testMain(self):
        extents = starsmashertools.helpers.extents.Extents(
            xmin = -0.5, xmax = 0.5,
            ymin = -1., ymax = 1.5,
            zmin = -0.2, zmax = 10.,
        )
        self.assertAlmostEqual(extents.min[0], -0.5)
        self.assertAlmostEqual(extents.min[1], -1.)
        self.assertAlmostEqual(extents.min[2], -0.2)
        self.assertAlmostEqual(extents.max[0], 0.5)
        self.assertAlmostEqual(extents.max[1], 1.5)
        self.assertAlmostEqual(extents.max[2], 10.)
        
        error = starsmashertools.helpers.extents.NegativeValueError
        with self.assertRaises(error): extents.width = -1
        with self.assertRaises(error): extents.breadth = -1
        with self.assertRaises(error): extents.height = -1
        with self.assertRaises(error): extents.volume = -1
        
        extents.width = 4.
        self.assertAlmostEqual(extents.width, 4.)
        self.assertAlmostEqual(extents.breadth, 2.5)
        self.assertAlmostEqual(extents.height, 10.2)
        self.assertAlmostEqual(extents.xmin, -2.)
        self.assertAlmostEqual(extents.xmax, 2.)

        extents.breadth = 5.
        self.assertAlmostEqual(extents.width, 4.)
        self.assertAlmostEqual(extents.breadth, 5.)
        self.assertAlmostEqual(extents.height, 10.2)
        self.assertAlmostEqual(extents.ymin, -2.25)
        self.assertAlmostEqual(extents.ymax, 2.75)

        extents.height = 3.
        self.assertAlmostEqual(extents.width, 4.)
        self.assertAlmostEqual(extents.breadth, 5.)
        self.assertAlmostEqual(extents.height, 3.)
        self.assertAlmostEqual(extents.zmin, 3.4)
        self.assertAlmostEqual(extents.zmax, 6.4)

        self.assertAlmostEqual(extents.volume, 60.)

        extents.volume = 1.
        self.assertAlmostEqual(extents.width, 1.0217459098580708088)
        self.assertAlmostEqual(extents.breadth,     1.2771823873225885110)
        self.assertAlmostEqual(extents.height,      0.7663094323935531066)
        self.assertAlmostEqual(extents.xmin,       -0.5108729549290354044)
        self.assertAlmostEqual(extents.xmax,        0.5108729549290354044)
        self.assertAlmostEqual(extents.ymin,       -0.3885911936612942555)
        self.assertAlmostEqual(extents.ymax,        0.8885911936612942555)
        self.assertAlmostEqual(extents.zmin,        4.5168452838032234467)
        self.assertAlmostEqual(extents.zmax,        5.2831547161967765533)
        
        extents.min = [0, 0, 0]
        extents.max = [2, 2, 2]

        self.assertAlmostEqual(extents.xmin, 0.)
        self.assertAlmostEqual(extents.ymin, 0.)
        self.assertAlmostEqual(extents.zmin, 0.)
        self.assertAlmostEqual(extents.xmax, 2.)
        self.assertAlmostEqual(extents.ymax, 2.)
        self.assertAlmostEqual(extents.zmax, 2.)
        
        self.assertAlmostEqual(extents.volume, 8.)
        
        extents.center = [5., 5., 5.]
        self.assertAlmostEqual(extents.xmin, 4.)
        self.assertAlmostEqual(extents.ymin, 4.)
        self.assertAlmostEqual(extents.zmin, 4.)
        self.assertAlmostEqual(extents.xmax, 6.)
        self.assertAlmostEqual(extents.ymax, 6.)
        self.assertAlmostEqual(extents.zmax, 6.)
        
        extents.center = [-5., -5., -5.]
        self.assertAlmostEqual(extents.xmin, -6.)
        self.assertAlmostEqual(extents.ymin, -6.)
        self.assertAlmostEqual(extents.zmin, -6.)
        self.assertAlmostEqual(extents.xmax, -4.)
        self.assertAlmostEqual(extents.ymax, -4.)
        self.assertAlmostEqual(extents.zmax, -4.)
        
        extents.xy = [0, 1, 0, 1]
        self.assertAlmostEqual(extents.xmin, 0.)
        self.assertAlmostEqual(extents.xmax, 1.)
        self.assertAlmostEqual(extents.ymin, 0.)
        self.assertAlmostEqual(extents.ymax, 1.)
        
        
        
if __name__ == '__main__':
    unittest.main(failfast = True)
