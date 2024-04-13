# Additional Matplotlib Artists for easier plotting
import matplotlib.collections
import matplotlib.artist
import matplotlib.axes
import numpy as np
import starsmashertools.helpers.argumentenforcer
import starsmashertools.preferences


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
        import starsmashertools.mpl.colorbar
        
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




class OutputPlot(object):
    """
    A :class:`matplotlib.artist.Artist` which represents the data contained in
    a :class:`~starsmashertools.lib.output.Output`.
    """
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def __init__(
            self,
            axes : matplotlib.axes.Axes,
            x : str,
            y : str,
            logx : bool = False,
            logy : bool = False,
            **kwargs
    ):
        import starsmashertools.mpl.axes
        
        self.x = x
        self.y = y
        self.logx = logx
        self.logy = logy
        self.kwargs = kwargs

        self._output = None

        # We need space for plotting non core particles and core particles for
        # the primary and secondary stars
        self.artists = []

        preferences = PlottingPreferences()

        primary_kwargs = next(preferences)
        primary_core_kwargs = preferences.get_core_preferences()
        primary_kwargs.update(self.kwargs)
        primary_core_kwargs.update(self.kwargs)
        
        secondary_kwargs = next(preferences)
        secondary_core_kwargs = preferences.get_core_preferences()
        secondary_kwargs.update(self.kwargs)
        secondary_core_kwargs.update(self.kwargs)
        
        self.artists = {
            'primary noncore' : axes.scatter(
                [np.nan], [np.nan], **primary_kwargs
            ),
            'primary core' : axes.scatter(
                [np.nan], [np.nan], **primary_core_kwargs
            ),
            'secondary noncore' : axes.scatter(
                [np.nan], [np.nan], **secondary_kwargs
            ),
            'secondary core' : axes.scatter(
                [np.nan], [np.nan], **secondary_core_kwargs
            ),
        }

        starsmashertools.mpl.axes.make_legend(axes)

        self._previous_simulation = None
        self._particles = {key:None for key in self.artists.keys()}
    
    def _reset(self):
        for key, artist in self.artists.items():
            artist.set_offsets(
                np.column_stack((
                    [np.nan, np.nan], [np.nan, np.nan]
                ))
            )
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def set_output(
            self,
            output : "starsmashertools.lib.output.Output",
    ):
        x = output[self.x]
        y = output[self.y]

        if self.logx: x = np.log10(x)
        if self.logy: y = np.log10(y)

        if output.simulation != self._previous_simulation:
            self._reset()
            
            binary = None
            if output.simulation.isBinary: binary = output.simulation
            elif output.simulation.isDynamical:
                try:
                    binary = output.simulation.get_children()[0]
                except: pass

            #IDs = np.arange(output['ntot'], dtype = int)
            #noncore = IDs[output['u'] != 0]
            #core = IDs[~noncore]

            core = output.get_core_particles()
            noncore = output.get_non_core_particles()
            
            self._particles['primary noncore'] = noncore
            self._particles['primary core'] = core
            self._particles['secondary noncore'] = np.asarray([])
            self._particles['secondary core'] = np.asarray([])

            if binary is not None:
                primary = binary.get_primary_IDs()
                secondary = binary.get_secondary_IDs()
                self._particles['primary noncore'] = primary[np.isin(primary, noncore)]
                self._particles['primary core'] = primary[np.isin(primary, core)]
                self._particles['secondary noncore'] = secondary[np.isin(secondary, noncore)]
                self._particles['secondary core'] = secondary[np.isin(secondary, core)]
            
            xy = np.column_stack((x, y))
            
            for key, IDs in self._particles.items():
                if len(IDs) == 0: continue
                self.artists[key].set_offsets(xy[IDs])
        else:
            for key, IDs in self._particles.items():
                artist = self.artists[key]
                
                offsets = self.artists[key].get_offsets()
                is_valid = np.any(np.isfinite(offsets))
                
                artist.set_visible(is_valid)
                
                legend = artist.axes.get_legend()
                for handle, text in zip(legend.legendHandles, legend.get_texts()):
                    if text.get_text() == artist.get_label():
                        handle.set_visible(is_valid)
                        if not is_valid: text.set_text("")
                        else: text.set_text(artist.get_label())
                
                if not is_valid: continue
                
                offsets[:,0] = x[IDs]
                offsets[:,1] = y[IDs]

        if self._previous_simulation is None:
            xmin, xmax = np.amin(xy[:,0]), np.amax(xy[:,0])
            ymin, ymax = np.amin(xy[:,1]), np.amax(xy[:,1])
            ax = self.artists['primary noncore'].axes
            xmargin, ymargin = ax.margins()
            dx = (xmax - xmin) * xmargin
            dy = (ymax - ymin) * ymargin
            ax.set_xlim(xmin - dx, xmax + dx)
            ax.set_ylim(ymin - dy, ymax + dy)
            
        self._previous_simulation = output.simulation

@starsmashertools.preferences.use
class PlottingPreferences(object):
    def __init__(self):
        import copy
        
        self._default = self.preferences.get('Plotting', 'defaults')
        if self._default is None: self._default = {}

        self.cores = self.preferences.get('Plotting', 'core kwargs')
        if self.cores is None: self.cores = []
        
        self._items = self.preferences.get('Plotting', 'kwargs')
        if self._items is None: self._items = []
        
        self._index = -1
    
    def __next__(self):
        import copy
        ret = copy.deepcopy(self._default)
        if self._index + 1 < len(self._items):
            self._index += 1
            ret.update(self._items[self._index])
        return ret

    def get_core_preferences(self):
        import copy
        ret = copy.deepcopy(self._default)
        if self._index < len(self._items): ret.update(self._items[self._index])
        if self._index < len(self.cores): ret.update(self.cores[self._index])
        return ret
