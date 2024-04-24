import unittest
import starsmashertools.helpers.argumentenforcer
import starsmashertools.lib.ray
import os
import basetest
import numpy as np

class TestRay(basetest.BaseTest):
    def testBasic(self):
        # Not allowed to have 0-direction
        with self.assertRaises(ValueError):
            ray = starsmashertools.lib.ray.Ray((0,0,0), (0,0,0))

        ray = starsmashertools.lib.ray.Ray((0,0,0), (1, 0, 0))
        self.assertTrue(isinstance(ray.position, np.ndarray))
        self.assertTrue(isinstance(ray.direction, np.ndarray))
        self.assertEqual(ray.position.shape, (3,))
        self.assertEqual(ray.direction.shape, (3,))
        self.assertEqual(ray.position[0], 0)
        self.assertEqual(ray.position[1], 0)
        self.assertEqual(ray.position[2], 0)
        self.assertEqual(ray.direction[0], 1)
        self.assertEqual(ray.direction[1], 0)
        self.assertEqual(ray.direction[2], 0)

        ray = starsmashertools.lib.ray.Ray((0,0,0), (10, 0, 0))
        self.assertEqual(ray.direction[0], 1)
        self.assertEqual(ray.direction[1], 0)
        self.assertEqual(ray.direction[2], 0)

        ray = starsmashertools.lib.ray.Ray((0,0,0), (0, 3, 4)) # 3,4,5 triangle
        self.assertAlmostEqual(ray.direction[0], 0.0)
        self.assertAlmostEqual(ray.direction[1], 0.6)
        self.assertAlmostEqual(ray.direction[2], 0.8)

        ray = starsmashertools.lib.ray.Ray((0,0,0), (3, 0, 4))
        self.assertAlmostEqual(ray.direction[0], 0.6)
        self.assertAlmostEqual(ray.direction[1], 0.0)
        self.assertAlmostEqual(ray.direction[2], 0.8)

        ray = starsmashertools.lib.ray.Ray((0,0,0), (3, 4, 0))
        self.assertAlmostEqual(ray.direction[0], 0.6)
        self.assertAlmostEqual(ray.direction[1], 0.8)
        self.assertAlmostEqual(ray.direction[2], 0.0)

        ray = starsmashertools.lib.ray.Ray((0,0,0), (4, 3, 0))
        self.assertAlmostEqual(ray.direction[0], 0.8)
        self.assertAlmostEqual(ray.direction[1], 0.6)
        self.assertAlmostEqual(ray.direction[2], 0.0)

        ray = starsmashertools.lib.ray.Ray((0,0,0), (4, 0, 3))
        self.assertAlmostEqual(ray.direction[0], 0.8)
        self.assertAlmostEqual(ray.direction[1], 0.0)
        self.assertAlmostEqual(ray.direction[2], 0.6)

    def testCastBasic(self):
        # Cast on +x
        ray = starsmashertools.lib.ray.Ray((0,0,0), (1, 0, 0))
        cast = ray.cast([1.], [0.], [0.], [0.5])
        self.assertEqual(cast.points.shape, (2,3))
        self.assertAlmostEqual(cast.points[0][0], 0.5)
        self.assertAlmostEqual(cast.points[0][1], 0.0)
        self.assertAlmostEqual(cast.points[0][2], 0.0)
        self.assertAlmostEqual(cast.points[1][0], 1.5)
        self.assertAlmostEqual(cast.points[1][1], 0.0)
        self.assertAlmostEqual(cast.points[1][2], 0.0)

        # Cast on -x
        ray = starsmashertools.lib.ray.Ray((0,0,0), (-1, 0, 0))
        cast = ray.cast([-1.], [0.], [0.], [0.5])
        self.assertEqual(cast.points.shape, (2,3))
        self.assertAlmostEqual(cast.points[0][0], -0.5)
        self.assertAlmostEqual(cast.points[0][1], 0.0)
        self.assertAlmostEqual(cast.points[0][2], 0.0)
        self.assertAlmostEqual(cast.points[1][0], -1.5)
        self.assertAlmostEqual(cast.points[1][1], 0.0)
        self.assertAlmostEqual(cast.points[1][2], 0.0)

        # Cast on +y
        ray = starsmashertools.lib.ray.Ray((0,0,0), (0, 1, 0))
        cast = ray.cast([0.], [1.], [0.], [0.5])
        self.assertEqual(cast.points.shape, (2,3))
        self.assertAlmostEqual(cast.points[0][0], 0.0)
        self.assertAlmostEqual(cast.points[0][1], 0.5)
        self.assertAlmostEqual(cast.points[0][2], 0.0)
        self.assertAlmostEqual(cast.points[1][0], 0.0)
        self.assertAlmostEqual(cast.points[1][1], 1.5)
        self.assertAlmostEqual(cast.points[1][2], 0.0)

        # Cast on -y
        ray = starsmashertools.lib.ray.Ray((0,0,0), (0, -1, 0))
        cast = ray.cast([0.], [-1.], [0.], [0.5])
        self.assertEqual(cast.points.shape, (2,3))
        self.assertAlmostEqual(cast.points[0][0], 0.0)
        self.assertAlmostEqual(cast.points[0][1], -0.5)
        self.assertAlmostEqual(cast.points[0][2], 0.0)
        self.assertAlmostEqual(cast.points[1][0], 0.0)
        self.assertAlmostEqual(cast.points[1][1], -1.5)
        self.assertAlmostEqual(cast.points[1][2], 0.0)

        # Cast on +z
        ray = starsmashertools.lib.ray.Ray((0,0,0), (0, 0, 1))
        cast = ray.cast([0.], [0.], [1.], [0.5])
        self.assertEqual(cast.points.shape, (2,3))
        self.assertAlmostEqual(cast.points[0][0], 0.0)
        self.assertAlmostEqual(cast.points[0][1], 0.0)
        self.assertAlmostEqual(cast.points[0][2], 0.5)
        self.assertAlmostEqual(cast.points[1][0], 0.0)
        self.assertAlmostEqual(cast.points[1][1], 0.0)
        self.assertAlmostEqual(cast.points[1][2], 1.5)

        # Cast on -z
        ray = starsmashertools.lib.ray.Ray((0,0,0), (0, 0, -1))
        cast = ray.cast([0.], [0.], [-1.], [0.5])
        self.assertEqual(cast.points.shape, (2,3))
        self.assertAlmostEqual(cast.points[0][0], 0.0)
        self.assertAlmostEqual(cast.points[0][1], 0.0)
        self.assertAlmostEqual(cast.points[0][2], -0.5)
        self.assertAlmostEqual(cast.points[1][0], 0.0)
        self.assertAlmostEqual(cast.points[1][1], 0.0)
        self.assertAlmostEqual(cast.points[1][2], -1.5)

        # Test bad values
        ray = starsmashertools.lib.ray.Ray((0,0,0), (1, 0, 0))
        cast = ray.cast([1.], [0.], [0.], [0.])
        self.assertEqual(cast.points.shape, (0,))
        cast = ray.cast([np.nan], [0.], [0.], [1.])
        self.assertEqual(cast.points.shape, (0,))
        cast = ray.cast([np.inf], [0.], [0.], [1.])
        self.assertEqual(cast.points.shape, (0,))

        cast = ray.cast([1., np.nan], [0., 0.], [0., 0.], [0.5, 0.5])
        self.assertEqual(cast.points.shape, (2,3))
        self.assertAlmostEqual(cast.points[0][0], 0.5)
        self.assertAlmostEqual(cast.points[0][1], 0.0)
        self.assertAlmostEqual(cast.points[0][2], 0.0)
        self.assertAlmostEqual(cast.points[1][0], 1.5)
        self.assertAlmostEqual(cast.points[1][1], 0.0)
        self.assertAlmostEqual(cast.points[1][2], 0.0)
        
        
    def testCastMultiple(self):
        # Cast on multiple spheres
        ray = starsmashertools.lib.ray.Ray((0,0,0), (1, 0, 0))
        cast = ray.cast([1., 2.], [0.,0.], [0.,0.], [0.1,0.1])
        self.assertEqual(cast.indices.shape, (4,))
        self.assertEqual(cast.indices[0], 0)
        self.assertEqual(cast.indices[1], 0)
        self.assertEqual(cast.indices[2], 1)
        self.assertEqual(cast.indices[3], 1)

        cast = ray.cast([1., 2.], [0.,0.], [0.,0.], [0.5, 1.5])
        self.assertEqual(cast.indices.shape, (4,))
        self.assertEqual(cast.indices[0], 0)
        self.assertEqual(cast.indices[1], 1)
        self.assertEqual(cast.indices[2], 0)
        self.assertEqual(cast.indices[3], 1)


if __name__ == "__main__":
    unittest.main(failfast=True)
