import starsmashertools.preferences
import starsmashertools.helpers.argumentenforcer
import numpy as np
import matplotlib.artist
import matplotlib.figure

def update_style_sheet_directories():
    """ Locate the style sheets listed in the preferences. """
    import starsmashertools
    import matplotlib.pyplot as plt

    directories = []
    try:
        directories = Figure.preferences.get('stylesheet directories')
    except: pass
    
    for directory in directories:
        directory = directory.format(**starsmashertools.__dict__)
        if directory in plt.style.core.USER_LIBRARY_PATHS: continue
        plt.style.core.USER_LIBRARY_PATHS += [directory]
    plt.style.core.reload_library()


@starsmashertools.preferences.use
class Figure(matplotlib.figure.Figure, object):
    """
    Used internally to create Matplotlib figures. This normalizes the process
    for creating figures to ensure consistency.
    """
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def __init__(
            self,
            *args,
            scale : list | tuple | np.ndarray = (1., 1.),
            bound : bool = True,
            debug : bool = False,
            **kwargs
    ):
        """
        Initializer. Review the Matplotlib documentation for 
        :func:`matplotlib.pyplot.subplots` for details which are not documented 
        here.
        
        Other Parameters
        ----------------
        *args
           Other positional arguments are passed directly to 
           :class:`matplotlib.figure.Figure`.

        scale : list | tuple | np.ndarray, default = (1., 1.)
           A 2-element iterable which scales the figure size. Values cannot be
           negative.

        bound : bool, default = True
           If `True`, artists will be forced to stay within the figure limits
           by adjusting the ``subplot_adjust`` properties.

        debug : bool, default = False
           If `True`, debug information will be drawn on the figure.

        **kwargs
           Other keyword arguments are passed directly to 
           :class:`matplotlib.figure.Figure`.
        """
        import matplotlib.patches
        
        scale = np.asarray(scale)
        idx = scale < 0
        if idx.any():
            raise ValueError("Keyword argument 'scale' cannot have negative values: " + str(scale))
        
        super(Figure, self).__init__(
            *args,
            **kwargs
        )
        self._original_size = self.get_size_inches()
        self._scale = np.asarray([1., 1.])
        self.scale = scale

        self._debug_bounds = matplotlib.patches.Rectangle(
            (0, 0),
            0, 0,
            clip_on = False,
            zorder = float('inf'),
            edgecolor = 'r',
            facecolor = 'none',
            lw = 1,
            ls = ':',
            transform = self.transFigure,
            visible = debug,
        )
        self.add_artist(self._debug_bounds)

        self.bound = bound
        self.debug = debug

    @property
    def debug(self): return self._debug_bounds.get_visible()

    @debug.setter
    def debug(self, value): self._debug_bounds.set_visible(value)

    @property
    def scale(self): return self._scale

    @scale.setter
    def scale(self, value):
        figsize = self._original_size * value
        super(Figure, self).set_size_inches(figsize)
        self._scale = value

    def set_size_inches(self, *args, **kwargs):
        ret = super(Figure, self).set_size_inches(*args, **kwargs)
        self._original_size = np.asarray(self.get_size_inches())
        return ret

    def draw(self, *args, **kwargs):
        if self.debug: # Update the debug box bounds
            ax_bounds = self.get_axes_bounds(renderer=self.canvas.get_renderer())
            xmin, xmax = float('inf'), -float('inf')
            ymin, ymax = float('inf'), -float('inf')
            for ax, bbox in ax_bounds.items():
                xmin = min(xmin, bbox.x0)
                xmax = max(xmax, bbox.x1)
                ymin = min(ymin, bbox.y0)
                ymax = max(ymax, bbox.y1)
            self._debug_bounds.set_bounds(xmin, ymin, xmax - xmin, ymax - ymin)
        super(Figure, self).draw(*args, **kwargs)
        if self.bound: self._fix_subplots_adjust()
        super(Figure, self).draw(*args, **kwargs)
    
    def show(self, *args, **kwargs):
        """ Display the figure without blocking the code execution if the CLI is
        being used. """
        import starsmashertools.bintools.cli
        import matplotlib.pyplot as plt
        
        plt.figure(num = self.number) # Focus this figure

        if starsmashertools.bintools.cli.CLI.instance is not None:
            # Using the CLI
            if not plt.isinteractive(): plt.ion()
            ret = plt.show(*args, **kwargs)
            plt.draw()
            plt.pause(0.001)
            return ret
        # Not using the CLI
        #return super(Figure, self).show(*args, **kwargs)
        return plt.show(*args, **kwargs)

    def savefig(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        plt.figure(num = self.number) # Focus this figure
        return super(Figure, self).savefig(*args, **kwargs)

    def subplots_adjust(self, **kwargs):
        # Force any "bound" axes to stay within the figure
        if self.bound:
            self._fix_subplots_adjust(**kwargs)
            self.canvas.draw_idle()
        else: super(Figure, self).subplots_adjust(**kwargs)
    
    def _fix_subplots_adjust(
            self,
            left = None,
            right = None,
            bottom = None,
            top = None,
            **kwargs
    ):
        if left is None: left = self.subplotpars.left
        if right is None: right = self.subplotpars.right
        if bottom is None: bottom = self.subplotpars.bottom
        if top is None: top = self.subplotpars.top

        for ax, bounds in self.get_axes_bounds().items():
            pos = ax.get_position()
            left = max(left, pos.x0 - bounds.x0)
            right = min(right, 1 - (bounds.x1 - pos.x1))
            bottom = max(bottom, pos.y0 - bounds.y0)
            top = min(top, 1 - (bounds.y1 - pos.y1))
        kwargs['left'] = left
        kwargs['right'] = right
        kwargs['bottom'] = bottom
        kwargs['top'] = top
        super(Figure, self).subplots_adjust(**kwargs)
    
    def get_axes_bounds(self, *args, **kwargs):
        import matplotlib.text
        import matplotlib.transforms
        ret = {}
        for ax in self.axes:
            bbox = ax.get_window_extent(*args, **kwargs)
            xmin, xmax, ymin, ymax = bbox.x0, bbox.x1, bbox.y0, bbox.y1
            for child in ax._get_all_children():
                if not child.get_visible(): continue
                ext = child.get_window_extent(*args, **kwargs)
                if ext.width <= 0 or ext.height <= 0: continue
                x0, x1, y0, y1 = ext.xmin, ext.xmax, ext.ymin, ext.ymax
                if isinstance(child, matplotlib.text.Text):
                    # The text can be wrongly sized by 1 pixel on any side, so here
                    # we err on the side of caution
                    text = child.get_text().strip()
                    if not text: continue
                    x0 -= 2
                    x1 += 2
                    y0 -= 2
                    y1 += 2

                xmin = min(xmin, x0)
                xmax = max(xmax, x1)
                ymin = min(ymin, y0)
                ymax = max(ymax, y1)
            ret[ax] = matplotlib.transforms.Bbox(
                [
                    [xmin, ymin],
                    [xmax, ymax],
                ],
            ).transformed(self.transFigure.inverted())
        return ret

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def save(
            self,
            path : str = 'figure.sstfig',
            extra : dict = {},
    ):
        import json
        import starsmashertools.helpers.file
        import starsmashertools.helpers.pickler
        data = {
            'figure' : starsmashertools.helpers.pickler.pickle_object(self),
        }
        for key, val in extra.items(): data[key] = val
        
        with starsmashertools.helpers.file.open(path, 'w') as f:
            json.dump(data, f)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @staticmethod
    def load(
            path : str = 'figure.sstfig',
    ):
        import json
        import starsmashertools.helpers.file
        import starsmashertools.helpers.pickler

        with starsmashertools.helpers.file.open(path, 'r') as f:
            data = json.load(f)
        figure = starsmashertools.helpers.pickler.unpickle_object(data['figure'])
        extra = {}
        for key, val in data.items():
            if key == 'figure': continue
            extra[key] = val

        # Ensure that the ColoredPlot artists have limits set correctly
        colored_plots = self.get_artists(
            artist_type = starsmashertools.mpl.artists.ColoredPlot,
        )
        handled_axes = {
            'x' : [],
            'y' : [],
        }
        for ax in self.axes:
            artists = [artist for artist in colored_plots if artist.axes is ax]
            if not artists: continue

            xys = [artist.get_xydata() for artist in artists]
            xys = [xy for xy in xys if len(xy) > 0]

            if ax not in handled_axes['x']:
                ax.set_xlim(
                    min([min(xy[:,0]) for xy in xys]),
                    max([max(xy[:,0]) for xy in xys]),
                )
                shared_x = ax.get_shared_x_axes()
                for a in self.axes:
                    if a not in shared_x: continue
                    if a in handled_axes['x']: continue
                    handled_axes['x'] += [a]
                if ax not in handled_axes['x']: handled_axes['x'] += [ax]
            
            if ax not in handled_axes['y']:
                ax.set_ylim(
                    min([min(xy[:,1]) for xy in xys]),
                    max([max(xy[:,1]) for xy in xys]),
                )
                shared_y = ax.get_shared_y_axes()
                for a in self.axes:
                    if a not in shared_y: continue
                    if a in handled_axes['y']: continue
                    handled_axes['y'] += [a]
                if ax not in handled_axes['y']: handled_axes['y'] += [ax]
            
        if extra: return figure, extra
        return figure

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get_artists(self, artist_type : type = matplotlib.artist.Artist):
        """ Obtain all artists of the given type. If the given type is None, 
        this returns all artists in the figure. """
        
        # Recursive get
        def get(artist, total = []):
            if not isinstance(artist, artist_type): return total
            if artist in total: return total
            total += [artist]
            if hasattr(artist, 'get_children'):
                for child in artist.get_children():
                    if child in total: continue
                    total = get(child, total = total)
            return total
        
        return get(self)


class PanelGridFigure(Figure, object):
    """
    This class represents a :class:`~.Figure` which has multiple axes, arranged
    as a grid of panels. Each axes is positioned such that 
    """
    def __init__(
            self,
    ):
        pass












# Store all the available Figure classes and subclasses for lookup later
import sys, inspect
class_names = {}
for name, _class in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    if issubclass(_class, Figure):
        _class.name = '.'.join([_class.__module__, _class.__qualname__])
        class_names[_class.name] = _class
