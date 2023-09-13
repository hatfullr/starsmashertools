import matplotlib.colorbar
import matplotlib.collections
import matplotlib.patches
import matplotlib.colors
import matplotlib.transforms
import numpy as np
import warnings
import starsmashertools.mpl.blackbodycmap
import starsmashertools.helpers.argumentenforcer

@starsmashertools.helpers.argumentenforcer.enforcetypes
def add_artist(
        colorbar : matplotlib.colorbar.Colorbar,
        artist : matplotlib.artist.Artist,
):
    def update_artist(*args, **kwargs):
        artist.set_norm(colorbar.norm)
        artist.set_cmap(colorbar.cmap)
    colorbar.ax.add_callback(update_artist)
    update_artist()


@starsmashertools.helpers.argumentenforcer.enforcetypes
def convert_to_blackbody(
        colorbar : matplotlib.colorbar.Colorbar,
        vmin : int | float = 1000,
        vmax : int | float = 40000,
):
    cmap = starsmashertools.mpl.blackbodycmap.get(vmin=vmin, vmax=vmax)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colorbar.cmap = cmap
    colorbar.norm = norm
    colorbar.draw_all()


# colorbar = the colorbar to change
# objects = the plotted objects which will be affected by the colorbar change
@starsmashertools.helpers.argumentenforcer.enforcetypes
def discretize(
        colorbar : matplotlib.colorbar.Colorbar,
        objects : list = [],
        nodraw : bool = False
):
    cax = colorbar.ax
    cmap = colorbar.cmap
    clim = cax.get_ylim()

    fig = cax.get_figure()
    figsize = fig.get_size_inches()
    
    # Update the figure (sometimes required?)
    if not nodraw: fig.canvas.draw()

    xaxis = cax.get_xaxis()
    yaxis = cax.get_yaxis()

    # Get ALL the ticks (major AND minor)
    ticks = {xaxis:[], yaxis:[]}
    lines = {xaxis:[], yaxis:[]}
    for axis in [xaxis, yaxis]:
        for tick in axis.get_major_ticks():
            line1 = tick.tick1line
            line2 = tick.tick2line
            if line1.get_visible() or line2.get_visible():
                ticks[axis] += [tick]
                if line1.get_visible(): lines[axis] += [line1]
                if line2.get_visible(): lines[axis] += [line2]
    
    # Ensure that there is only 1 axis on the colorbar
    if not (ticks[xaxis] or ticks[yaxis]):
        raise Exception("To discretize a colorbar it must have major ticks on exactly 1 of its axes")

    # Remove the axis that we aren't using
    if ticks[yaxis]: axis = yaxis
    else: axis = xaxis
    ticks = ticks[axis]
    lines = lines[axis]

    # Convert to NumPy arrays for easier handling
    lines = np.asarray(lines)


    
    patches = []

    # Create the patches that are used when the colorbar is 'extended',
    # which means little triangles are added to the top and/or bottom.
    if colorbar.extend != 'neither':
        extendfrac = 0.05 # From the docs
        if colorbar.extendfrac is not None: extendfrac = colorbar.extendfrac
        if colorbar.extend in ['both', 'min']:
            verts = np.array([[0.5,-extendfrac],[0.,0.],[1.,0.]])
            patches.append(matplotlib.patches.Polygon(
                verts,
                facecolor=cmap(0.),
            ))
        if colorbar.extend in ['both','max']:
            verts = np.array([[0.5,1.+extendfrac],[0.,1.],[1.,1.]])
            patches.append(matplotlib.patches.Polygon(
                verts,
                facecolor=cmap(1.),
            ))
    
    
    # Sort the lines by their positions
    positions = []
    for line in lines:
        lw = line.get_linewidth() / 72.
        bbox = matplotlib.transforms.Bbox([[0, 0], [lw, lw]])
        bbox = bbox.transformed(fig.dpi_scale_trans).transformed(cax.transData.inverted())
        width = abs(bbox.width)
        height = abs(bbox.height)
        if axis is xaxis:
            positions += [line.get_xdata()[0] + 2*width]
        else:
            positions += [line.get_ydata()[0] + 2*height]
    
    positions = np.asarray(sorted(positions))

    if axis is xaxis:
        xy = lambda i: (positions[i], 0)
        width = lambda i: positions[i+1] - positions[i]
        height = lambda i: 1.
    else:
        xy = lambda i: (0, positions[i])
        width = lambda i: 1.
        height = lambda i: positions[i+1] - positions[i]

    color = lambda i: cmap((positions[i] - clim[0]) / (clim[1] - clim[0]))

    # Create the patches that will cover over the colorbar to give the illusion
    # of a discretized colorbar

    for i in range(len(positions)-1):
        patches.append(matplotlib.patches.Rectangle(
            xy(i),
            width(i),
            height(i),
            color=color(i),
        ))
    
    p = matplotlib.collections.PatchCollection(
        patches,
        match_original=True,
    )
    cax.add_collection(p)







    # Now modify the objects that rely on this colorbar




    

    # https://stackoverflow.com/a/45178154/4954083
    def get_value_from_cm(color, cmap, colrange=[0.,1.]):
        color=to_rgb(color)
        r = np.linspace(colrange[0],colrange[1], 256)
        norm = Normalize(colrange[0],colrange[1])
        mapvals = cmap(norm(r))[:,:3]
        distance = np.sum((mapvals - color)**2, axis=1)
        return r[np.argmin(distance)]
    
    if len(objects) != 0:
        # Flatten the objects if it is an array
        # https://stackoverflow.com/a/12474246/4954083
        flatten=lambda l: sum(map(flatten,l),[]) if isinstance(l,(list,tuple,np.ndarray)) else [l]
        if isinstance(objects,(np.ndarray,list,tuple)): objects = flatten(objects)
        # Turn it into an iterable otherwise
        else: objects = [objects]

    #if colorbar.mappable not in objects:
    #    objects += [colorbar.mappable]

    for i, obj in enumerate(objects):
        objproperties = obj.properties()
        
        if isinstance(obj, matplotlib.lines.Line2D):
            color = obj.get_color()
            
            # Check to see if c contains just 1 color. If so, make c iterable
            if not any([isinstance(it,(tuple,list,np.ndarray)) for it in color]): color = [color]

            mod_positions = (positions - clim[0]) / (clim[1] - clim[0])
            
            cmap_locs = np.array([get_value_from_cm(c, cmap) for c in color])
            valid = np.isfinite(cmap_locs)
            below = np.logical_and(valid, cmap_locs < min(mod_positions))
            above = np.logical_and(valid, cmap_locs >= max(mod_positions))
            between = np.logical_and(~below, ~above)
            if np.any(below): cmap_locs[below] = clim[0]
            if np.any(above): cmap_locs[above] = clim[1]
            if np.any(between):
                for k in range(len(mod_positions) - 1):
                    smaller = min(mod_positions[k], mod_positions[k + 1])
                    larger = max(mod_positions[k], mod_positions[k + 1])
                    idx = np.logical_and(smaller <= cmap_locs[between], cmap_locs[between] < larger)
                    if np.any(idx):
                        cmap_locs[between][idx] = smaller
            
            color = [cmap(cmap_loc) for cmap_loc in cmap_locs]
            if len(color) == 1: color = color[0]
            objects[i].set_markerfacecolor(color)

        elif 'array' in objproperties.keys() and objproperties['array'] is not None:
            # In this case, the colors are stored in the actual image data
            arr = objproperties['array']
            
            if isinstance(obj, matplotlib.image.AxesImage):
                warnings.warn("imshow data will be edited", stacklevel=2)
                data = arr.data
            elif isinstance(obj, matplotlib.collections.PathCollection):
                data = arr.data.ravel()
            else:
                raise NotImplementedError("Object type '%s'" % type(obj).__name__)

            valid = np.isfinite(data)
            below = np.logical_and(valid, data < min(positions))
            above = np.logical_and(valid, data >= max(positions))
            between = np.logical_and(~below, ~above)
            if np.any(below): data[below] = clim[0]
            if np.any(above): data[above] = clim[1]
            if np.any(between):
                for k in range(len(positions) - 1):
                    smaller = min(positions[k], positions[k + 1])
                    larger = max(positions[k], positions[k + 1])
                    idx = np.logical_and(smaller <= data[between], data[between] < larger)
                    if np.any(idx):
                        data[between][idx] = smaller

            obj.set_array(arr)





