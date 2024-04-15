import starsmashertools.preferences
import matplotlib.figure
import matplotlib
import matplotlib.pyplot as plt
import starsmashertools.helpers.argumentenforcer
import numpy as np
import matplotlib.artist

def update_style_sheet_directories():
    """ Locate the style sheets listed in the preferences. """
    import starsmashertools

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
            **kwargs
    ):
        """
        Initializer. Review the Matplotlib documentation for 
        ``matplotlib.pyplot.subplots`` for details which are not documented 
        here.
        
        Other Parameters
        ----------------
        *args
           Other positional arguments are passed directly to 
           ``matplotlib.figure.Figure``.

        scale : list | tuple | np.ndarray, default = (1., 1.)
           A 2-element iterable which scales the figure size. Values cannot be
           negative.

        **kwargs
           Other keyword arguments are passed directly to 
           ``matplotlib.figure.Figure``.
        """
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

    @property
    def scale(self): return self._scale

    @scale.setter
    def scale(self, value):
        figsize = self._original_size * value
        super(Figure, self).set_size_inches(figsize)
        self._scale = value

    #def close(self):
    #    plt.pause(1)
    #    plt.close(fig = self)

    def set_size_inches(self, *args, **kwargs):
        ret = super(Figure, self).set_size_inches(*args, **kwargs)
        self._original_size = np.asarray(self.get_size_inches())
        return ret

    def show(self, *args, **kwargs):
        """ Display the figure without blocking the code execution if the CLI is
        being used. """
        import starsmashertools.bintools.cli
        
        plt.figure(num = self.number) # Focus this figure

        if starsmashertools.bintools.cli.CLI.instance is not None:
            # Using the CLI
            if not plt.isinteractive(): plt.ion()
            ret = plt.show(*args, **kwargs)
            plt.draw()
            plt.pause(0.001)
            return ret
        # Not using the CLI
        return super(Figure, self).show(*args, **kwargs)
        #return plt.show(*args, **kwargs)

    def savefig(self, *args, **kwargs):
        plt.figure(num = self.number) # Focus this figure
        return super(Figure, self).savefig(*args, **kwargs) #plt.savefig(*args, **kwargs)

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

                
        

def subplots(
        *args,
        FigureClass = Figure,
        style : str = 'starsmashertools',
        **kwargs
):
    update_style_sheet_directories()
    plt.style.use(style)
    return plt.subplots(*args, FigureClass = Figure, **kwargs)






