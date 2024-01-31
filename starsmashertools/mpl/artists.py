# Additional Matplotlib Artists for easier plotting
import matplotlib.collections
import starsmashertools.mpl.colorbar
import numpy as np


class ColoredPlot(matplotlib.collections.LineCollection, object):
    def __init__(
            self,
            axes,
            x, y,
            colors='C0',
            colorbar=None,
            joinstyle='round',
            capstyle='round',
            **kwargs
    ):
        segments = list(zip(zip(x[:-1], y[:-1]), zip(x[1:], y[1:])))

        if isinstance(colors, str) and not hasattr(colors, "__iter__"):
            kwargs['colors'] = colors

        #if isinstance(colors, np.ndarray):
        #    self.colors = np.repeat(colors, 2)
            #self.colors = np.asarray(list(zip(colors[:-1], colors[1:])))
        
        super(ColoredPlot, self).__init__(
            segments,
            joinstyle=joinstyle,
            capstyle=capstyle,
            **kwargs
        )
        
        if not isinstance(colors, str) and hasattr(colors, "__iter__"):
            if len(colors) - 1 == len(segments):
                self.set_array(colors)
        axes.add_collection(self)
        
        if colorbar is not None:
            if not isinstance(colors, str) and hasattr(colors, "__iter__"):
                if len(colors) - 1 == len(segments):
                    starsmashertools.mpl.colorbar.add_artist(colorbar, self)

        axes._request_autoscale_view("x")
        axes._request_autoscale_view("y")

    def get_xydata(self):
        import numpy as np
        segments = self.get_segments()
        xy = []
        for segment in segments:
            for _xy in segment:
                xy += [_xy]
        return np.asarray(xy)
        
    def get_colors(self):
        import numpy as np
        segments = self.get_segments()
        c = self.get_array()
        colors = []
        for i, segment in enumerate(segments):
            for _xy in segment:
                colors += [c[i]]
        return np.asarray(colors)
