import unittest
import os
import numpy as np
import basetest
import warnings

@unittest.skipIf(__name__ != '__main__', 'matplotlib tests can only be run on the main thread')
class TestAxes(basetest.BaseTest):
    def setUp(self):
        import matplotlib.pyplot as plt
        plt.close('all')

    def tearDown(self):
        import matplotlib.pyplot as plt
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
        import matplotlib.pyplot as plt
        import starsmashertools.mpl
        import string
        fig, ax = starsmashertools.mpl.subplots(debug = True)
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
    unittest.main(failfast=True)
