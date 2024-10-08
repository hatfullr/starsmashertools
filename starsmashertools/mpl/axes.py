# This contains shortcut methods for working on matplotlib Axes objects.
import matplotlib.axes
import matplotlib.transforms
import matplotlib.colorbar
import matplotlib.path
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.projections
import matplotlib.text
import numpy as np
import starsmashertools.helpers.argumentenforcer
import copy

@starsmashertools.helpers.argumentenforcer.enforcetypes
def get_resolution(axes : matplotlib.axes._axes.Axes):
    # Return the pixel resolution of the axes
    pos = axes.get_position().transformed(axes.transAxes)
    return int(pos.width), int(pos.height)

@starsmashertools.helpers.argumentenforcer.enforcetypes
def make_legend(axes : matplotlib.axes._axes.Axes):
    import starsmashertools.mpl.artists
    legend_kwargs = {}
    try:
        legend_kwargs = starsmashertools.mpl.artists.PlottingPreferences.preferences.get('legend')
    except: pass
    
    markersize = legend_kwargs.pop('markersize', None)
    legend = axes.legend(**legend_kwargs)
    if markersize is not None:
        try:
            for handle in legend.legendHandles:
                handle.set_sizes([markersize])
        except AttributeError:
            # Matplotlib 3.7+ apparently doesn't work like this anymore
            pass

@starsmashertools.helpers.argumentenforcer.enforcetypes
def colorbar(
        axes : matplotlib.axes._axes.Axes | np.ndarray | list | tuple,
        padding : int | float = 0.01,
        width : float = 0.05,
        orientation : str = 'vertical',
        location : str = 'right',
        **kwargs
):
    starsmashertools.helpers.argumentenforcer.enforcevalues({
        'orientation' : ['vertical', 'horizontal'],
        'location' : ['right', 'left', 'top', 'bottom'],
    })

    if not isinstance(axes, np.ndarray):
        axes = np.asarray(axes, dtype=matplotlib.axes._axes.Axes)

    fig = axes.flatten()[0].get_figure()
    cax = fig.add_axes([0,0,0,0])
    def set_cax_position(*args, **kwargs):
        bbox = get_bbox(axes)

        if orientation == 'vertical':
            y0, y1 = bbox.y0, bbox.y1
            if location == 'right' : x0 = bbox.x1 + padding
            elif location == 'left' : x0 = bbox.x0 - width - padding
            else: raise ValueError("When 'orientation' is 'vertical', 'location' must be either 'right' or 'left', not '%s'" % location)
            x1 = x0 + width
        else:
            x0, x1 = bbox.x0, bbox.x1
            if location == 'top' : y0 = bbox.y1 + padding
            elif location == 'bottom' : y0 = bbox.y0 - width - padding
            else: raise ValueError("When 'orientation' is 'horizontal', 'location' must be either 'top' or 'bottom', not '%s'" % location)
            y1 = y0 + width
        cax.set_position([x0, y0, x1 - x0, y1 - y0])

    fig.canvas.mpl_connect('draw_event', set_cax_position)
    return matplotlib.colorbar.Colorbar(cax, **kwargs)

    

# Returns a bbox that encapsulates all the given axes
@starsmashertools.helpers.argumentenforcer.enforcetypes
def get_bbox(ax : matplotlib.axes._axes.Axes | np.ndarray | list | tuple):
    if not isinstance(ax, np.ndarray):
        ax = np.asarray(ax, dtype=matplotlib.axes._axes.Axes)
    #if isinstance(ax, matplotlib.axes._axes.Axes):
    #    ax = np.array(ax, dtype = matplotlib.axes._axes.Axes)
    x0, x1 = 1., 0.
    y0, y1 = 1., 0.
    for a in ax.flatten():
        pos = a.get_position()
        x0, x1 = min(x0, pos.x0), max(x1, pos.x1)
        y0, y1 = min(y0, pos.y0), max(y1, pos.y1)
    return matplotlib.transforms.Bbox([[x0, y0], [x1, y1]])
    

# Use a power law to set the x or y axis scale of an Axes or
@starsmashertools.helpers.argumentenforcer.enforcetypes
def set_tickscale_power_law(
        ax : matplotlib.axes._axes.Axes,
        power : int | float,
        which : str = 'both',
):

    starsmashertools.helpers.argumentenforcer.enforcevalues({
        'which' : ['x', 'y', 'both'],
    })
    
    scale = lambda axis: matplotlib.scale.FuncScale(
        axis,
        (
            lambda arr: np.sign(arr) * np.abs(arr)**power,
            lambda arr: np.sign(arr) * np.abs(arr)**(1./power),
        ),
    )
    
    def update_ticks(ax):
        if ax.stale: ax.get_figure().canvas.draw_idle()
        if which in ['both', 'x']:
            # Make sure there are ticks to affect in the plot
            xticks = ax.xaxis.get_majorticklabels()
            if xticks:
                xlim = ax.get_xlim()
                for xtick in xticks:
                    if ((xlim[0] <= xtick._x and xtick._x <= xlim[1]) and
                        xtick.get_text()):
                        # Set the scale and update the margins so that
                        # the plot looks proper
                        ax.set_xscale(scale(ax))
                        xmargin, ymargin = ax.margins()
                        xmargin = np.sign(xmargin) * abs(xmargin)**power
                        ax.margins(xmargin, ymargin)
                        break
            
        if which in ['both', 'y']:
            yticks = ax.yaxis.get_majorticklabels()
            if yticks:
                ylim = ax.get_ylim()
                for ytick in yticks:
                    if ((ylim[0] <= ytick._y and xtick._y <= ylim[1]) and
                        ytick.get_text()):
                        ax.set_yscale(scale(ax))
                        xmargin, ymargin = ax.margins()
                        ymargin = np.sign(ymargin) * abs(ymargin)**power
                        ax.margins(xmargin, ymargin)
                        break

    fig = ax.get_figure()
    fig.canvas.mpl_connect("draw_event", lambda *args, **kwargs: update_ticks(ax))


def text(
        ax : matplotlib.axes._axes.Axes,
        string : str,
        position : list[float, float],
        **kwargs
):
    return ax.annotate(
        string,
        position,
        xycoords='axes fraction',
        **kwargs
    )


class Axes(matplotlib.axes.Axes, object):
    name = 'Axes'
    
    def _get_all_children(self):
        def get(obj, children = []):
            if obj not in children: children += [obj]
            if hasattr(obj, 'get_children'):
                for child in obj.get_children():
                    for c in get(child, children = children):
                        if c not in children: children += [c]
            return children

        children = []
        for child in self.get_children():
            children = get(child, children = children)
        return children


