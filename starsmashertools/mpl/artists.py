# Additional Matplotlib Artists for easier plotting
import starsmashertools.preferences
from starsmashertools.preferences import Pref
import matplotlib.collections
import matplotlib.axes
import numpy as np
import starsmashertools.helpers.argumentenforcer
from starsmashertools.helpers.apidecorator import api
import starsmashertools.lib.flux
import starsmashertools.lib.output
import copy


@starsmashertools.preferences.use
class PlottingPreferences(object):
    def __init__(self):
        self._default = self.preferences.get('defaults')
        if self._default is None: self._default = {}

        self.cores = self.preferences.get('core kwargs')
        if self.cores is None: self.cores = []
        
        self._items = self.preferences.get('kwargs')
        if self._items is None: self._items = []
        
        self._index = -1
    
    def __next__(self):
        ret = copy.deepcopy(self._default)
        if self._index + 1 < len(self._items):
            self._index += 1
            ret.update(self._items[self._index])
        return ret

    def get_core_preferences(self):
        ret = copy.deepcopy(self._default)
        if self._index < len(self._items): ret.update(self._items[self._index])
        if self._index < len(self.cores): ret.update(self.cores[self._index])
        return ret


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
    r"""
    A class which represents the data contained in a 
    :class:`~starsmashertools.lib.output.Output` for plotting.
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
    def set_output(self, output : starsmashertools.lib.output.Output):
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
                try:
                    handles = legend.legendHandles
                except AttributeError: # Matplotlib 3.7+
                    handles = legend.legend_handles
                for handle, text in zip(handles, legend.get_texts()):
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
class FluxPlot(object):
    r"""
    A class for easily performing Matplotlib plotting operations on a 
    :class:`~.lib.flux.FluxResult`\.
    """
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def __init__(
            self,
            axes : matplotlib.axes.Axes,
            result : starsmashertools.lib.flux.FluxResult,
    ):
        self._axes = axes
        self._result = result
        self._image = None
        self._highlight = None
        self._outline = None
    
    @api
    def remove(self):
        r"""
        Remove all artists this object is responsible for from its axes.
        """
        artists = [
            self._image,
            self._highlight,
            self._outline,
        ]
        for artist in artists:
            if artist is not None:
                try:
                    artist.remove()
                except: pass

        self._image = None
        self._highlight = None
        self._outline = None

    @api
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def imshow(
            self,
            data : np.ndarray,
            extent : list | tuple | np.ndarray | type(None) = None,
            **kwargs
    ):
        r"""
        Plot a quantity on a Matplotlib :class:`matplotlib.axes.Axes`\.
        
        Parameters
        ----------
        data : :class:`numpy.ndarray`
            The image to plot.

        Other Parameters
        ----------------
        extent : list, tuple, :class:`numpy.ndarray`\, None, default = None
            See :meth:`matplotlib.axes.Axes.imshow`\. If `None`\, then the 
            extents from the flux results are used.

        kwargs
            Other keyword arguments are passed directly to 
            :meth:`matplotlib.axes.Axes.imshow`\. Note, keyword ``origin`` is
            always be set to ``'lower'`` regardless of any value it has in
            ``kwargs``\. If keywords appear in the preferences, they will be 
            used unless specified here.

        Returns
        -------
        :class:`matplotlib.axes.AxesImage`
        """
        import matplotlib.pyplot as plt
        import starsmashertools.helpers.warnings

        if self._image is not None:
            raise Exception("An AxesImage has already been created for this FluxPlot instance: %s" % str(self._image))

        try:
            prefs = self.preferences.get('images')
        except: prefs = {}
        
        prefs.update(kwargs)
        kwargs = prefs
        
        if extent is None: extent = self._result['image']['extent']
        
        kwargs['origin'] = 'lower'
        self._image = self._axes.imshow(
            np.swapaxes(data, 0, 1),
            extent = extent,
            **kwargs
        )
        return self._image
    
    @api
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def highlight_particles(
            self,
            IDs : list | tuple | np.ndarray,
            **kwargs
    ):
        r"""
        There must be an image plotted with :meth:`~.imshow` first.

        Parameters
        ----------
        IDs : list, tuple, :class:`numpy.ndarray`
            The particle IDs to highlight. IDs which are not in the
            ``contributing_IDs`` list of the :class:`~.lib.flux.FluxResult` are
            excluded.
        
        Other Parameters
        ----------------
        kwargs
            Other keyword arguments are passed directly to 
            :meth:`matplotlib.axes.Axes.imshow`\.

        Returns
        -------
        :class:`matplotlib.axes.AxesImage`
        """
        import matplotlib.colors
        
        if self._image is None:
            raise Exception("An image must be plotted first with method imshow()")
        if self._highlight is not None:
            raise Exception("Cannot highlight more than once. Please remove the old highlight first.")

        if not isinstance(IDs, np.ndarray): IDs = np.asarray(IDs)
        
        
        try:
            prefs = self.preferences.get('particle highlight')
        except: prefs = {}

        prefs.update(kwargs)
        kwargs = prefs
        kwargs['origin'] = 'lower'
        
        output = copy.deepcopy(self._result.output)
        output.rotate(
            xangle = self._result['kwargs']['theta'],
            zangle = self._result['kwargs']['phi'],
        )

        contributors = self._result['particles']['contributing_IDs']
        IDs = IDs[np.isin(IDs, contributors, assume_unique = True)]
        
        rloc = np.full(output['ntot'], np.nan)        
        rloc[contributors] = self._result['particles']['rloc']

        x = output['x'][IDs]
        y = output['y'][IDs]
        z = output['z'][IDs]
        rloc = rloc[IDs]
        
        data = np.full(self._image.get_array().shape, False, dtype = bool)
        
        ntot = len(rloc)
        resolution = data.shape
        xmin, xmax, ymin, ymax = self._result['image']['extent']
        dx, dy = self._result['image']['dx'], self._result['image']['dy']

        ijloc_arr = np.column_stack((
            ((x - xmin) / dx).astype(int), # iloc
            ((y - ymin) / dy).astype(int), # jloc
        ))

        ijminmax_arr = np.column_stack((
            # imin
            np.maximum(
                ((x - rloc - xmin) / dx).astype(int), # same as floor
                np.zeros(ntot, dtype = int),
            ),
            # imax
            np.minimum(
                ((x + rloc - xmin) / dx).astype(int) + 1, # same as ceil
                np.full(ntot, resolution[0], dtype = int),
            ),
            # jmin
            np.maximum(
                ((y - rloc - ymin) / dy).astype(int), # same as floor
                np.zeros(ntot, dtype = int),
            ),
            # jmax
            np.minimum(
                ((y + rloc - ymin) / dy).astype(int) + 1, # same as ceil
                np.full(ntot, resolution[1], dtype = int),
            ),
        ))

        # Omit ijloc that are outside the image
        inside_image = np.logical_and(
            np.logical_and(ijloc_arr[:,0] >= 0, ijloc_arr[:,0] <= resolution[0]),
            np.logical_and(ijloc_arr[:,1] >= 0, ijloc_arr[:,1] <= resolution[1]),
        )
        if inside_image.any():
            for i, (iloc, jloc) in enumerate(ijloc_arr):
                if inside_image[i]:
                    imin, imax, jmin, jmax = ijminmax_arr[i]
                    xloc_arr = xmin + dx * np.arange(imin, imax)
                    yloc_arr = ymin + dy * np.arange(jmin, jmax)
                    deltax2_arr = (xloc_arr - x[i])**2
                    dist2_arr = deltax2_arr[:, None] + (yloc_arr - y[i])**2
                    idx = dist2_arr <= rloc[i]**2
                    if not idx.any(): continue
                    data[imin:imax, jmin:jmax][idx] = True
            
        if 'color' in kwargs.keys():
            color = matplotlib.colors.to_rgba(kwargs['color'])
            _data = np.full((data.shape[0], data.shape[1], 4), np.nan)
            _data[data] = color
            data = _data
            kwargs.pop('color')
            
        self._highlight = self._axes.imshow(
            np.swapaxes(data, 0, 1),
            extent = self._image.get_extent(),
            **kwargs
        )
        return self._highlight

    @api
    def highlight_dusty_particles(self, **kwargs):
        T_dust_min, T_dust = self._result['kwargs']['dust_Trange']
        contributors = self._result['particles']['contributing_IDs']
        IDs = np.arange(self._result.output['ntot'])
        dusty = starsmashertools.lib.flux.find_dusty_particles(
            self._result.output,
            T_dust_min = T_dust_min,
            T_dust_max = T_dust,
            kappa_dust = self._result['kwargs']['dust_opacity'],
        )
        idx = np.logical_and(
            dusty,
            np.isin(IDs, contributors, assume_unique = True),
        )
        return self.highlight_particles(IDs[idx], **kwargs)

    @api
    def outline_particles(
            self,
            IDs : list | tuple | np.ndarray,
            **kwargs
    ):
        r"""
        There must be an image plotted with :meth:`~.imshow` first.

        Parameters
        ----------
        IDs : list, tuple, :class:`numpy.ndarray`
            The particle IDs to outline. IDs which are not in the
            ``contributing_IDs`` list of the :class:`~.lib.flux.FluxResult` are
            excluded.
        
        Other Parameters
        ----------------
        kwargs
            Other keyword arguments are passed directly to 
            :meth:`matplotlib.collections.EllipseCollection.__init__`\.

        Returns
        -------
        :class:`matplotlib.collections.EllipseCollection`
        """
        import matplotlib.collections
        
        if self._image is None:
            raise Exception("An image must be plotted first with method imshow()")
        if self._outline is not None:
            raise Exception("Cannot outline more than once. Please remove the old outline first.")

        if not isinstance(IDs, np.ndarray): IDs = np.asarray(IDs)
        
        try:
            prefs = self.preferences.get('particle outline')
        except: prefs = {}

        prefs.update(kwargs)
        kwargs = prefs

        output = copy.deepcopy(self._result.output)
        output.rotate(
            xangle = self._result['kwargs']['theta'],
            zangle = self._result['kwargs']['phi'],
        )

        contributors = self._result['particles']['contributing_IDs']
        IDs = IDs[np.isin(IDs, contributors, assume_unique = True)]
        
        rloc = np.full(output['ntot'], np.nan)        
        rloc[contributors] = self._result['particles']['rloc']

        x = output['x'][IDs]
        y = output['y'][IDs]
        rloc = rloc[IDs]

        ext = self._axes.get_window_extent(renderer = self._axes.get_figure().canvas.get_renderer())
        aspect = ext.width / ext.height

        if aspect < 1:
            width = 2 * rloc
            height = width / aspect
        else:
            height = 2 * rloc
            width = height * aspect

        self._outline = self._axes.add_collection(
            matplotlib.collections.EllipseCollection(
                width, height,
                offsets = np.column_stack((x, y)),
                offset_transform = self._axes.transData,
                units = 'x',
                #units = 'xy',
                angles = np.zeros(x.shape),
                **kwargs
            )
        )
        return self._outline

    @api
    def outline_dusty_particles(self, **kwargs):
        T_dust_min, T_dust = self._result['kwargs']['dust_Trange']
        contributors = self._result['particles']['contributing_IDs']
        IDs = np.arange(self._result.output['ntot'])
        dusty = starsmashertools.lib.flux.find_dusty_particles(
            self._result.output,
            T_dust_min = T_dust_min,
            T_dust_max = T_dust,
            kappa_dust = self._result['kwargs']['dust_opacity'],
        )
        idx = np.logical_and(
            dusty,
            np.isin(IDs, contributors, assume_unique = True),
        )
        return self.outline_particles(IDs[idx], **kwargs)
