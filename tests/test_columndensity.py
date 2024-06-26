import unittest
import basetest
import os
import starsmashertools

curdir = os.getcwd()

class TestColumnDensity(basetest.BaseTest):
    def test(self): pass
    
    """ # Uncomment to see a plot. Keep commented when you're done
    def test_plot(self):
        import matplotlib.pyplot as plt
        import starsmashertools.mpl.image.columndensity
        simulation = starsmashertools.get_simulation(os.path.join(curdir, 'flux'))
        output = simulation.get_output(0)
        
        fig, ax = plt.subplots()
        colden = starsmashertools.mpl.image.columndensity.ColumnDensity(
            output,
            'rho',
            ax,
            max_resolution = (100, 100),
            vmin = 0,
            vmax = 5,
            log = True,
            parallel = True,
            #parallel = False,
        )

        plt.show()

    """

class TestLoader(unittest.TestLoader, object):
    def getTestCaseNames(self, *args, **kwargs):
        return [
            #'test_plot',
        ]

if __name__ == '__main__':
    import inspect
    import re
    import starsmashertools.mpl.image.columndensity
    
    comment_checks = [
        # Remove # comments first
        re.compile("(?<!['\"])#.*", flags = re.M),
        # Then remove block comments (which can be commented out by #)
        re.compile('(?<!\')(?<!\\\\)""".*?"""', flags = re.M | re.S),
        re.compile("(?<!\")(?<!\\\\)'''.*?'''", flags = re.M | re.S),
    ]
    
    src = inspect.getsource(starsmashertools.mpl.image.columndensity)

    # Remove all comments
    for check in comment_checks:
        for match in check.findall(src):
            src = src.replace(match, '')
    
    if '@profile' in src:
        loader = TestLoader()
        suite = unittest.TestSuite()
        for name in loader.getTestCaseNames():
            suite.addTest(TestColumnDensity(name))
        runner = unittest.TextTestRunner()
        runner.run(suite)
    else:
        # This is the normal method
        unittest.main(failfast=True, testLoader=TestLoader())

