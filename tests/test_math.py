# Testing *some* of the math functions, mainly EffGravPot
import unittest
import starsmashertools.math
import numpy as np
import basetest
import typing

class TestEffGravPot(basetest.BaseTest):
    def setUp(self):
        # This has a COM at (0, 0, 0)
        self.g = starsmashertools.math.EffGravPot(
            # don't change these
            [ 1.0,  0.5], # m1 =  1.0, m2 =  m1/2
            [-1.0,  2.0], # x1 = -1.0, x2 =  2.0
            [ 0.0,  0.0], # y1 =  0.0, y2 =  0.0
            [ 0.0,  0.0], # z1 =  0.0, z2 =  0.0
            period = 2 * np.pi, # makes omega = 1
            G = 1.,
        )
    
    def test_get(self):
        xyz = np.array([
            self.g._com,
            self.g._com + np.array([0, 1, 0]),
        ])
        expected = np.array([
            # At the COM, centripetal acceleration is 0
            -1.25,
            # Now y = 1
            -1/np.sqrt(2) - 1/(2*np.sqrt(5)) - 0.5,
        ])

        for i,v in enumerate(self.g.get(xyz)):
            self.assertAlmostEqual(
                v,
                expected[i],
                msg = str(i)+" "+str(v)+" != " + str(expected[i]),
            )

    def test_get_gradient(self):
        xyz = np.array([
            self.g._com,
            self.g._com + np.array([0, 1, 0]),
        ])
        # Remember, expected values are \nabla\varphi, not -\nabla\varphi
        expected = np.array([
            # At the COM, grav. force is only in x direction, and centripetal
            # acceleration is 0
            [-(0.5/2.0**2 - 1.0/(-1.0)**2), 0, 0],
            # Now y = 1, so grav. force is in x and y directions and centripetal
            # acceleration != 0. Distance to m1 is sqrt(2), and direction is
            # [-0.707, -0.707, 0] (it's normalized). Distance to m2 is
            # sqrt(2^2 + 1^2) = sqrt(5), in direction [1/sqrt(5), -1/sqrt(5), 0]
            [np.nan, np.nan, np.nan], # Fill in below
        ])
        # Through pen and paper, we find
        g1 = 1/(2*np.sqrt(2)) * np.array([-1, -1, 0])
        g2 = 1/(10*np.sqrt(5)) * np.array([2, -1, 0])
        a = np.array([0, -1, 0])
        expected[1] = -(g1 + g2 + a) # \nabla\varphi = -a

        for i,v in enumerate(self.g.get_gradient(xyz)):
            self.assertTrue(
                np.array_equal(v, expected[i]),
                msg = str(i)+" "+str(v)+" != " + str(expected[i]),
            )


    def test_lagrange(self):
        plot = False
        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
        
            n = 10000
            xyz = np.column_stack((
                np.linspace(-5, 4, n),
                np.zeros(n),
                np.zeros(n),
            ))
            phi = self.g.get(xyz)
            ax.plot(xyz[:,0], phi)
            ax.set_ylim(-8, -1)
            ax.set_xlim(-4, 4)
            
            ax.axvline(
                self.g._com[0],
                color = 'k',
                linestyle = '--',
                label = 'COM',
            )

        expected = [
            np.array([ 0.35035035, 0., 0.]),
            np.array([ 2.45245245, 0., 0.]),
            np.array([-1.75175175, 0., 0.]),
        ]
        
        for i, point in enumerate(self.g.get_lagrange(which = [1,2,3])):
            self.assertTrue(
                np.allclose(point, expected[i]),
                msg=str(i)+" "+str(point)+" != " + str(expected[i]),
            )
            if plot:
                ax.axvline(
                    point[0],
                    color = ax._get_lines.get_next_color(),
                    label = 'L%d' % (i + 1),
                )
        
        if plot:
            ax.legend()
            plt.show()
    
    
    def test_lagrange_scales(self):
        import starsmashertools.lib.units

        plot = False

        Rsun = float(starsmashertools.lib.units.Unit(1,'Rsun').convert('cm'))
        Msun = float(starsmashertools.lib.units.Unit(1,'Msun').convert('g'))
        
        g = starsmashertools.math.EffGravPot(
            np.asarray(self.g.m) * Msun,
            *(np.asarray(self.g.xyz.T) * Rsun),
            period = float(starsmashertools.lib.units.Unit(1, 'day').convert('s')),
        )

        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            
            n = 10000
            sep = g.x[1] - g.x[0]
            xyz = np.column_stack((
                np.linspace(g.x[0] - 2*sep, g.x[1] + 2*sep, n),
                np.zeros(n),
                np.zeros(n),
            ))
            phi = g.get(xyz)
            ax.plot(xyz[:,0] / Rsun, -np.log10(-phi))
            
            ax.axvline(
                self.g._com[0] / Rsun,
                color = 'k',
                linestyle = '--',
                label = 'COM',
            )

        expected = [
            np.array([ 5.19418149e+10, 0., 0.]),
            np.array([ 3.68346749e+11, 0., 0.]),
            np.array([-3.56112827e+11, 0., 0.]),
        ]
        
        for i, point in enumerate(g.get_lagrange(which = [1,2,3])):
            self.assertTrue(np.allclose(point, expected[i], rtol=1.e-2), msg=str(i)+" "+str(point)+" != " + str(expected[i]))
            if plot:
                ax.axvline(
                    point[0] / Rsun,
                    color = ax._get_lines.get_next_color(),
                    label = 'L%d' % (i + 1),
                )

        if plot:
            ax.legend()
            plt.show()





class TestIntegral(basetest.BaseTest):
    def test(self):
        # A simple test for several different types of functions
        functions = {
            'x^2 from 0 to 3' : {
                'func' : lambda x: x**2,
                'lower' : 0, 'upper' : 3, 'expected' : 9,
                'places' : 5,
            },
            'x^2 from -3 to 0' : {
                'func' : lambda x: x**2,
                'lower' : -3, 'upper' : 0, 'expected' : 9,
                'places' : 5,
            },
            'x^2 from 4 to 7' : {
                'func' : lambda x: x**2,
                'lower' : 4, 'upper' : 7, 'expected' : 93,
                'places' : 5,
            },
            '1/x from 1 to 2' : {
                'func' : lambda x: 1/x,
                'lower' : 1, 'upper' : 2, 'expected' : np.log(2),
                'places' : 5,
            },
            '1/x from -2 to -1' : {
                'func' : lambda x: 1/x,
                'lower' : -2, 'upper' : -1, 'expected' : -np.log(2),
                'places' : 5,
            },
            'sin(x) from 0 to pi' : {
                'func' : lambda x: np.sin(x),
                'lower' : 0, 'upper' : np.pi, 'expected' : 2,
                'places' : 5,
            },
            # A strange function
            'x sin(x^2 cos(16x)) from 0 to 2' : {
                'func' : lambda x: x * np.sin(x**2 * np.cos(16 * x)),
                'lower' : 0, 'upper' : 2, 'expected' : -0.0642987942,
                'places' : 3,
            },
        }
        for _class in starsmashertools.math.Integral.__subclasses__():
            for name, obj in functions.items():
                if isinstance(obj['expected'], typing.Callable):
                    expected = obj['expected'](obj['upper']) - obj['expected'](obj['lower'])
                else: expected = obj['expected']
                self.assertAlmostEqual(
                    _class(obj['func'], obj['lower'], obj['upper'])(np.linspace(obj['lower'], obj['upper'], 1000)),
                    expected,
                    places = obj['places'],
                    msg = "{_class:s} {func:s}".format(
                        _class = str(_class),
                        func = name,
                    ),
                )
        


class Test_GandV(basetest.BaseTest):
    def test_G(self):
        # Ensure self-consistency with G and V. We expect
        # G(x,h) = V(4h - |x-h|, h)
        import starsmashertools.lib.kernels

        """ Uncomment to show a debug plot
        import matplotlib.pyplot as plt
        for kernel in starsmashertools.lib.kernels._BaseKernel.__subclasses__():
            if kernel.name == 'uniform': continue
            h = 1
            x = np.linspace(0, 2*h, 1000)
            plt.plot(
                x,
                starsmashertools.math.G(x, h, kernel()),
                label = kernel.__name__,
            )
            
        plt.gca().legend()
        plt.show()
        """

        for kernel in starsmashertools.lib.kernels._BaseKernel.__subclasses__():
            if kernel.name == 'uniform': continue

            self.assertAlmostEqual(
                starsmashertools.math.G(1, 1, kernel()),
                1,
                msg = str(kernel),
                places = 5,
            )
            self.assertAlmostEqual(
                starsmashertools.math.G(2, 1, kernel()),
                0,
                msg = str(kernel),
                places = 5,
            )

            x = np.linspace(1 - 0.1, 1 + 0.1, 10)
            for value in starsmashertools.math.G(x, 1, kernel()):
                self.assertAlmostEqual(value, 1, msg = str(kernel), places = 5)

        self.assertEqual(starsmashertools.math.G(0, 1, gflag = 1), 1)
        self.assertEqual(starsmashertools.math.G(0, 1, gflag = 0), 0)

    def test_dG(self):
        self.assertEqual(starsmashertools.math.dG(0, 1, gflag = 1), 0)
        self.assertEqual(starsmashertools.math.dG(0, 1, gflag = 0), 0)
        self.assertNotEqual(starsmashertools.math.dG(0.1, 1, gflag = 0), 0)
        self.assertEqual(starsmashertools.math.dG(0.1, 1, gflag = 1), 0)

        """ Uncomment to show a debug plot
        import matplotlib.pyplot as plt
        for kernel in starsmashertools.lib.kernels._BaseKernel.__subclasses__():
            if kernel.name == 'uniform': continue
            h = 1
            x = np.linspace(0, 2*h, 1000)
            plt.plot(
                x,
                starsmashertools.math.dG(x, h, kernel(), gflag = 0),
                label = kernel.__name__,
            )
            
        plt.gca().legend()
        plt.show()
        """
        
    
if __name__ == "__main__":
    unittest.main(failfast=True)
        
