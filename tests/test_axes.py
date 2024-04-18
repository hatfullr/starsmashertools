import unittest
import os
import starsmashertools.mpl.axes
import starsmashertools.mpl.figure
import starsmashertools.mpl.debug
import numpy as np
import basetest
import warnings

try:
    import matplotlib.pyplot as plt
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

if not has_matplotlib:
    warnings.warn("No Matplotlib installation detected. Test '%s' will not be run." % os.path.basename(__file__))

if has_matplotlib:
    class TestAxes(basetest.BaseTest):
        figsize = (8, 8)
            
        def tearDown(self):
            # Check if the test failed
            errors = []
            for testmethod, stuff in self._outcome.errors:
                if stuff is not None:
                    errors += [[testmethod, stuff[1]]]
                    break
            if errors:
                for i, error in enumerate(errors):
                    if isinstance(error[1], AssertionError):
                        print("Error %d, %s: %s" % (i, error[0], error[1]))
                    else: raise(error[1])
                plt.show()
            plt.close('all')
        
        def testIsBound(self):
            import string
            fig, ax = starsmashertools.mpl.figure.subplots(debug = True)
            plt.subplots_adjust(
                left = 0,
                right = 1,
                top = 1,
                bottom = 0,
            )
            s = string.ascii_lowercase+string.ascii_uppercase
            ax.set_xlabel(s)
            ax.set_ylabel(s)
            ax.set_title(s)

            fig.canvas.draw()
            
            self.assertGreater(round(fig.subplotpars.left, 6), 0, msg = 'left')
            self.assertLess(round(fig.subplotpars.right,6), 1, msg = 'right')
            self.assertGreater(round(fig.subplotpars.bottom,6), 0, msg = 'bottom')
            self.assertLess(round(fig.subplotpars.top, 6), 1, msg = 'top')

            for ax, bounds in fig.get_axes_bounds().items():
                self.assertGreaterEqual(round(bounds.x0,6), 0, msg = 'x0')
                self.assertLessEqual(round(bounds.x1,6), 1, msg = 'x1')
                self.assertGreaterEqual(round(bounds.y0,6), 0, msg = 'y0')
                self.assertLessEqual(round(bounds.y1,6), 1, msg = 'y1')




if __name__ == "__main__":
    if has_matplotlib:
        unittest.main(failfast=True)
