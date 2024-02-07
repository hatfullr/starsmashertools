# Some helpful functions for debugging annoying Matplotlib issues
import matplotlib.artist
import starsmashertools.helpers.argumentenforcer
import matplotlib.collections
import matplotlib.lines
import matplotlib.patches
import matplotlib.patheffects
import matplotlib.transforms
import numpy as np

default_zorder = float('inf')


class Outline(matplotlib.artist.Artist, object):
    allowed_types = [
        matplotlib.collections.LineCollection,
        matplotlib.lines.Line2D,
    ]
    
    def __init__(
            self,
            artist,
            transform = None,
            facecolor = 'none',
            edgecolor = 'r',
            show_extent : bool = False,
            outlinewidth : float | int | np.float_ | np.integer = 3, # in points
            **kwargs
    ):
        super(Outline, self).__init__()

        self._artist_type = None
        for _type in Outline.allowed_types:
            if isinstance(artist, _type) or issubclass(artist.__class__, _type):
                self._artist_type = _type
                break
        else:
            raise NotImplementedError("Artist of type '%s' is not supported" % type(artist).__name__)
        
        self.artist = artist
        
        kwargs['clip_on'] = False

        self._rect = matplotlib.patches.Rectangle(
            (0, 0),
            0, 0,
            transform = self.artist.get_figure().transFigure,
            facecolor = facecolor,
            edgecolor = edgecolor,
            **kwargs
        )

        self.artist.set_path_effects([
            matplotlib.patheffects.Stroke(linewidth=outlinewidth, foreground=edgecolor),
            matplotlib.patheffects.Normal(),
        ])
        
        fig = self.artist.get_figure()
        fig.add_artist(self)
        fig.add_artist(self._rect)

        self.show_extent = show_extent

    @property
    def show_extent(self): return self._rect.get_visible()

    @show_extent.setter
    def show_extent(self, value): self._rect.set_visible(value)

    def remove(self, *args, **kwargs):
        self._rect.remove(*args, **kwargs)

    def get_extent(self, renderer = None):
        if renderer is None: renderer = self.get_figure().canvas.get_renderer()
        
        if self._artist_type == matplotlib.collections.LineCollection:
            # Obtain the data limits
            segments = self.artist.get_segments()
            if not segments: return matplotlib.transforms.Bbox.null()
                
            datalim = [[float('inf'), float('inf')], [-float('inf'), -float('inf')]]
            for segment in segments:
                for x, y in segment:
                    if np.isfinite(x):
                        datalim[0][0] = min(datalim[0][0], x)
                        datalim[1][0] = max(datalim[1][0], x)
                    if np.isfinite(y):
                        datalim[0][1] = min(datalim[0][1], y)
                        datalim[1][1] = max(datalim[1][1], y)
                
            return matplotlib.transforms.Bbox(
                # data space
                datalim
            ).transformed(
                # data to display space
                self.artist.axes.transData
            )
            
        return self.artist.get_window_extent(renderer = renderer)

    def set_visible(self, *args, **kwargs):
        super(Outline, self).set_visible(*args, **kwargs)
        self._rect.set_visible(self.get_visible() and self.show_extent)

    def get_bbox(self):
        fig = self.get_figure()
        extent = self.get_extent(
            renderer = fig.canvas.get_renderer(),
        )
        return matplotlib.transforms.Bbox(extent).transformed(
            fig.transFigure.inverted()
        )
        
    def draw(self, renderer):
        if not self.get_visible(): return

        # Update the Bbox
        bbox = self.get_bbox()
        self._rect.set_bounds(bbox.x0, bbox.y0, bbox.width, bbox.height)
        
        super(Outline, self).draw(renderer)


@starsmashertools.helpers.argumentenforcer.enforcetypes
def outline(
        artist : matplotlib.artist.Artist,
        **kwargs
):
    return Outline(artist, **kwargs)
    

@starsmashertools.helpers.argumentenforcer.enforcetypes
def draw_extent(
        artist : matplotlib.artist.Artist,
        clip_on : bool = False,
        zorder : float = default_zorder,
        **kwargs
):
    artist.get_figure().canvas.draw()

    fig = artist.get_figure()

    # Ensure that the figure is up-to-date
    fig.canvas.draw()

    extent = artist.get_window_extent(
        renderer = fig.canvas.get_renderer(),
    )
    bbox = matplotlib.transforms.Bbox(extent)
    kwargs['transform'] = None
    
    bbox_artist = matplotlib.patches.Rectangle(
        (bbox.x0, bbox.y0),
        bbox.width,
        bbox.height,
        clip_on = clip_on,
        zorder = zorder,
        **kwargs
    )
    fig.add_artist(bbox_artist)
    return bbox_artist

