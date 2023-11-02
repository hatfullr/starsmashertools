# General convenience methods for working with Matplotlib

import numpy as np
import matplotlib.transforms
import matplotlib.text
import matplotlib.axes
import starsmashertools.mpl.axes

def _global_label(axes, text, loc, offset=(0, 0), anchor='figure', **kwargs):
    if not isinstance(axes, matplotlib.figure.Figure):
        if isinstance(axes, np.ndarray): fig = axes.flatten()[0].get_figure()
        else: fig = axes.get_figure()
    else:
        _temp = axes
        axes = axes.axes
        fig = _temp
    
    bbox = starsmashertools.mpl.axes.get_bbox(axes)
    
    center = (
        0.5 * (bbox.x0 + bbox.x1),
        0.5 * (bbox.y0 + bbox.y1),
    )
    
    if loc == 'left':
        if anchor == 'figure': x = offset[0]
        elif anchor == 'axes':
            x = bbox.x0 - offset[0]
            kwargs['ha'] = 'left'
        else: raise NotImplementedError(anchor)
        y = center[1] + offset[1]
    elif loc == 'right':
        if anchor == 'figure': x = 1. - offset[0]
        elif anchor == 'axes':
            x = bbox.x1 + offset[0]
            kwargs['ha'] = 'right'
        else: raise NotImplementedError(anchor)
        y = center[1] + offset[1]
    elif loc == 'bottom':
        if anchor == 'figure': y = offset[1]
        elif anchor == 'axes':
            y = bbox.y0 - offset[1]
            kwargs['va'] = 'top'
        else: raise NotImplementedError(anchor)
        x = center[0] + offset[0]
    elif loc == 'top':
        if anchor == 'figure': y = 1. - offset[1]
        elif anchor == 'axes':
            y = bbox.y1 + offset[1]
            kwargs['va'] = 'bottom'
        else: raise NotImplementedError(anchor)
        x = center[0] + offset[0]
    else:
        raise ValueError("Argument 'loc' must be one of 'bottom' or 'top' for x labels and 'left' or 'right' for y labels, not '%s'" % loc)
    
    kwargs['transform'] = kwargs.get('transform', fig.transFigure)
    kwargs['figure'] = fig

    text = matplotlib.text.Text(
        x = x,
        y = y,
        text = text,
        **kwargs
    )
    fig.artists.append(text)

    # Matplotlib usually places its text just slightly too low, outside
    # what it thinks is the text's window extent. We correct for that here.
    x, y = text.get_position()
    ha = text.get_horizontalalignment()
    va = text.get_verticalalignment()
    bump = 0.001
    if ha == 'left': x += bump
    elif ha == 'right': x -= bump
    if va == 'bottom': y += bump
    elif va == 'top' : y -= bump
    text.set_position((x,y))

    
    # You can use the commented lines below to draw a debug box around
    # the text
    #import matplotlib.patches
    #fig.canvas.draw()
    #extent = text.get_window_extent(renderer=fig.canvas.get_renderer())
    #extent = extent.transformed(fig.transFigure.inverted())
    #box = matplotlib.patches.Rectangle(
    #    (extent.x0, extent.y0),
    #    extent.width,
    #    extent.height,
    #    transform = fig.transFigure,
    #    figure = fig,
    #    clip_on = False,
    #    fill=False,
    #    color='r',
    #    linewidth=0.1,
    #)
    #fig.artists.append(box)
    
    return text
    
    

# Make a single x label on the figure
def global_xlabel(fig, text, offset=(0, 0), loc='bottom', va='bottom', ha='center', **kwargs):
    return _global_label(
        fig,
        text,
        loc,
        offset = offset,
        va = va,
        ha = ha,
        **kwargs
    )

# Make a single y label on the figure
def global_ylabel(fig, text, offset=(0, 0), loc='left', va='center', ha='left', rotation='vertical', **kwargs):
    return _global_label(
        fig,
        text,
        loc,
        offset = offset,
        va = va,
        ha = ha,
        rotation = rotation,
        **kwargs
    )


    
