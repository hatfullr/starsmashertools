import unittest
import basetest
import starsmashertools.lib.kernels
import numpy as np
import inspect

def get_all_functions():
    for name, member in inspect.getmembers(starsmashertools.lib.kernels):
        if not inspect.isclass(member): continue
        if member is starsmashertools.lib.kernels._BaseKernel: continue
        if not issubclass(member, starsmashertools.lib.kernels._BaseKernel):
            continue
        yield member

class TestKernelFunction(basetest.BaseTest):
    def test_numerics(self):
        for function in get_all_functions():
            for compact_support in 10**np.linspace(-10, 100, 111):

                f = function(compact_support = compact_support)
                fzero = f.scaled(0, 1)

                # Check consistency on vast numeric scales
                for h in 10**np.linspace(-10, 100, 111):
                    # All functions must return 0 outside the kernel
                    edge = compact_support * h
                    outside = edge + 1.e-6 * edge
                    self.assertEqual(
                        0,
                        f.scaled(outside, h),
                        msg = 'outside kernel != 0 for %s' % function.__name__,
                    )
                    self.assertAlmostEqual(
                        np.log10(fzero),
                        np.log10(f.scaled(0, h)),
                        msg = 'f(0, h) != f(0, 1) for %s' % function.__name__,
                    )

    def test_tabulated(self):
        h = 1
        compact_support = 2
        for function in get_all_functions():
            f1 = function(tabulated = False, compact_support = compact_support)
            f2 = function(tabulated = True, compact_support = compact_support)
            self.assertTrue(f2.tabulated)
            x = np.linspace(0, compact_support * h, 100)
            for _x in x:
                self.assertAlmostEqual(f1(_x, h), f2(_x, h), places = 5)

    def test_integrated(self):
        import matplotlib.pyplot as plt
        compact_support = 2
        for function in get_all_functions():
            f = function(integrated = True, compact_support = compact_support)
            f2 = function(integrated = False, compact_support = f.compact_support)

            self.assertEqual(0, f(f.compact_support + 1.e-6, 1.))
            self.assertEqual(0, f.scaled(f.compact_support + 1.e-6, 1.))

            self.assertGreater(
                f.scaled(0, 1),
                f2.scaled(0, 1),
                msg = function.__name__,
            )
            
            x = np.linspace(0, f.compact_support, 100)
            l = plt.plot(
                x / f.compact_support,
                np.array([f(_x, 1) for _x in x]),
                label = function.__name__,
            )[0]

            plt.plot(
                x / f.compact_support,
                np.array([f2(_x, 1) for _x in x]),
                color = l.get_color(),
                linestyle = '--',
            )
            
        plt.legend()
        #plt.show() # Uncomment to see results

        plt.close('all')
            
    def test_get_by_name(self):
        for function in get_all_functions():
            self.assertIs(
                starsmashertools.lib.kernels.get_by_name(function.name),
                function,
                msg = function.__name__,
            )

    """ # Uncomment to see a plot
    def test_cubic_spline(self):
        import matplotlib.pyplot as plt
        compact_support = 2
        h = 1.
        
        f = starsmashertools.lib.kernels.CubicSplineKernel(
            compact_support = compact_support
        )

        x = np.linspace(0, compact_support * h, 100)
        y = np.full(x.shape, np.nan)
        
        for i in range(len(x)):
            y[i] = f(x[i], h)

        plt.plot(x / (compact_support * h), y)
        plt.show()
    """

    """ # Uncomment to see a plot
    def test_wendlandc4(self):
        import matplotlib.pyplot as plt
        compact_support = 2
        h = 1.
        
        f = starsmashertools.lib.kernels.WendlandC4Kernel(
            compact_support = compact_support
        )
        
        x = np.linspace(0, compact_support * h, 100)
        y = np.full(x.shape, np.nan)
        
        for i in range(len(x)):
            y[i] = f(x[i], h)

        plt.plot(x / (compact_support * h), y)
        plt.show()
    """
    


if __name__ == '__main__':
    unittest.main(failfast = True)
