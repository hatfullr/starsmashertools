import unittest
import starsmashertools
import starsmashertools.flux.fluxfinder
import os
import numpy as np
import sys

T_dust = 1000.0
T_dust_min = 100.0
kappa_dust = 1.0
do_dust = 1

c_speed=2.99792458e10
a_const=7.565767e-15

tau_skip = 1.e-5
tau_s = 20

colorbar_xbuffer = 0.005
colorbar_width = 0.015
axis_space = 0.075

def read_flux(filename):
    with open(filename, 'r') as f:
        xmin, ymin, dx, dy, Nx, Ny = f.readline().strip().split()
        xmin, ymin = float(xmin), float(ymin)
        dx, dy = float(dx), float(dy)
        Nx, Ny = int(Nx), int(Ny)

        data = np.full((Nx, Ny), None, dtype=object)
        for i, line in enumerate(f):
            data[i] = line.strip().split()
    data = data.astype(float)
    return data, xmin, ymin, dx, dy

class TestFlux(unittest.TestCase):
    def setUp(self):
        directory = os.path.join(
            starsmashertools.SOURCE_DIRECTORY,
            'tests',
            'flux_test',
        )
        self.simulation = starsmashertools.get_simulation(directory)

    def plot(self, result, expected, extent1, extent2):
        import matplotlib.pyplot as plt
        import matplotlib
        import warnings

        warnings.filterwarnings(action='ignore')

        logresult = np.log10(result)
        logexpected = np.log10(expected)
        
        idx1 = np.isfinite(logresult)
        idx2 = np.isfinite(logexpected)
        
        vmin = min(
            np.nanmin(logresult[idx1]),
            np.nanmin(logexpected[idx2]),
        )
        vmax = max(
            np.nanmax(logresult[idx1]),
            np.nanmax(logexpected[idx2]),
        )
        
        figsize = list(matplotlib.rcParams['figure.figsize'])
        figsize[0] *= 2.5
        fig, ax = plt.subplots(ncols = 3, figsize=figsize, sharex=True, sharey=True)
        plt.subplots_adjust(
            left=0.01,
            wspace=0.,
            right=0.864,
            top = 0.935,
            bottom=0.029,
        )

        for a in np.asarray(ax).flatten():
            a.tick_params(
                which='both',
                axis='both',
                left=False,
                right=False,
                top=False,
                bottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False,
                labelbottom=False,
            )


        xmin = min(extent1[0], extent2[0])
        xmax = max(extent1[1], extent2[1])
        ymin = min(extent1[2], extent2[2])
        ymax = max(extent1[3], extent2[3])

        extent = [xmin, xmax, ymin, ymax]

        ax[0].set_title('result')
        ax[1].set_title('expected')
        ax[2].set_title('difference')
        
        im0 = ax[0].imshow(
            logresult,
            extent = extent,
            interpolation = 'none',
            aspect = 'auto',
            vmin = vmin,
            vmax = vmax,
        )

        im1 = ax[1].imshow(
            logexpected,
            extent = extent,
            interpolation = 'none',
            aspect = 'auto',
            vmin = vmin,
            vmax = vmax,
        )

        p = np.log10(np.abs((result - expected)/expected))
        im2 = ax[2].imshow(
            p,
            extent = extent,
            interpolation = 'none',
            aspect = 'auto',
            vmin = np.nanmin(p[np.isfinite(p)]),
            vmax = np.nanmax(p[np.isfinite(p)]),
        )

        pos = ax[1].get_position()
        cax1 = fig.add_axes([pos.x1 + colorbar_xbuffer, pos.y0, colorbar_width, pos.height])
        colorbar1 = fig.colorbar(im1, cax=cax1)
        colorbar1.set_label(r'$\log_{10}(\mathrm{Flux})$')

        pos = ax[2].get_position()
        cax2 = fig.add_axes([pos.x1 + colorbar_xbuffer, pos.y0, colorbar_width, pos.height])
        colorbar2 = fig.colorbar(im2, cax=cax2)
        colorbar2.set_label(r'$\log_{10}(\mathrm{Absolute\ relative\ difference})$')

        def onUpdate(ax, cax1, cax2):
            pos = ax[1].get_position()
            cax1.set_position([pos.x1 + colorbar_xbuffer, pos.y0, colorbar_width, pos.height])

            ax[2].set_position([pos.x1 + axis_space, pos.y0, pos.width, pos.height])

            pos = ax[2].get_position()
            cax2.set_position([pos.x1 + colorbar_xbuffer, pos.y0, colorbar_width, pos.height])

            fig = np.asarray(ax).flatten()[0].get_figure()
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("draw_event", lambda *args,ax=ax, cax1=cax1, cax2=cax2, **kwargs: onUpdate(ax, cax1, cax2))
        
        warnings.resetwarnings()
        
        plt.show()
    
    def test(self, plot = False):
        import warnings
        warnings.filterwarnings(action = 'ignore')
        
        output = self.simulation.get_output()

        expected, expected_xmin, expected_ymin, expected_dx, expected_dy = read_flux(
            output.path.replace('.sph', '.surf_br')
        )
        expected_xmax = expected_xmin + expected_dx * (expected.shape[0] - 1)
        expected_ymax = expected_ymin + expected_dy * (expected.shape[1] - 1)
        extent = [expected_xmin, expected_xmax, expected_ymin, expected_ymax]
        
            
        finder = starsmashertools.flux.fluxfinder.FluxFinder(
            output,
            resolution = (np.asarray(expected.shape) - 1).tolist(),
            tau_ray_max = tau_s,
            viewing_angle = (0, 0, 0),
            extent = extent,
        )

        warnings.resetwarnings()
        
        result = finder.get()
        if np.any(result < 0):
            raise Exception("Negative flux found")

        e = None
        try:
            self.assertEqual(result.shape, expected.shape)
            self.assertTrue(np.allclose(result, expected, equal_nan=True))
        except Exception as _e:
            e = _e

        if plot:
            expected_extent = [
                expected_xmin,
                expected_xmin + expected_dx * expected.shape[0],
                expected_ymin,
                expected_ymin + expected_dy * expected.shape[1],
            ]
            self.plot(result, expected, finder.extent, expected_extent)

        if e: raise(e)


if __name__ == "__main__":
    import sys
    if '--plot' in sys.argv:
        t = TestFlux()
        t.setUp()
        t.test(plot=True)
    else:
        unittest.main(failfast=True)


