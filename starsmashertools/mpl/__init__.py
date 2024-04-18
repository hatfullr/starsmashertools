import typing
from starsmashertools.helpers.apidecorator import api
import starsmashertools.helpers.argumentenforcer
import matplotlib
import matplotlib.figure
import matplotlib.axes

# Prefer using the Tk backend (seems it's faster)
matplotlib.use('tkagg', force = False)

@starsmashertools.helpers.argumentenforcer.enforcetypes
@api
def subplots(
        *args,
        FigureClass : str | typing.Type[matplotlib.figure.Figure] | type(None) = 'starsmashertools.mpl.figure.Figure',
        AxesClass : str | typing.Type[matplotlib.axes.Axes] | type(None) = 'starsmashertools.mpl.axes.Axes',
        style : str | type(None) = 'starsmashertools',
        subplot_kw : dict | type(None) = None,
        **kwargs
):
    """
    Similar to :func:`matplotlib.pyplot.subplots`, except allows for the 
    use of custom :class:`matplotlib.figure.Figure` and 
    :class:`matplotlib.axes.Axes` classes.

    Parameters
    ----------
    args
        Positional arguments are passed directly to 
        :func:`matplotlib.pyplot.subplots`.

    Other Parameters
    ----------------
    FigureClass : str, Figure class, None, default = 'starsmashertools.mpl.figure.Figure'
        The kind of :class:`matplotlib.figure.Figure` to create. If a `str` is 
        given, it is an identifier which should match a ``name`` property of one
        of the custom Figure subclasses in :mod:`~.mpl.figure`. If a Figure 
        class is given, it is used. If `None`, the regular 
        :class:`matplotlib.figure.Figure` class is used.

    AxesClass : str, Axes class, None, default = 'starsmashertools.mpl.axes.Axes'
        The kind of :class:`matplotlib.axes.Axes` to create. If a `str` is 
        given, it is an identifier which should match a ``name`` property of one
        of the custom Axes subclasses in :mod:`~.mpl.figure`. If an Axes class 
        is given, it is used. If `None`, the regular 
        :class:`matplotlib.axes.Axes` class is used. If you provide your own
        class, it must be registered as a projection with Matplotlib first.

    style : str, None, default = 'starsmashertools'
        The name of a Matplotlib stylesheet to use. If `None` then no stylesheet
        is used.

    subplot_kw : dict, None, defualt = None
        The same as keyword ``subplot_kw`` in 
        :func:`matplotlib.pyplot.subplots`, except the key ``'projection'`` will
        always be set to the value of keyword argument ``AxesClass`` if it is 
        not `None`. If this keyword is `None` and ``AxesClass`` is not `None`, 
        then a new ``subplot_kw`` :py:class:`dict` will be created.

    kwargs
        Other keyword arguments are passed directly to 
        :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`

    ax : Matplotlib Axes or array of Axes
        The class provided by ``AxesClass`` will be used for each.
    """
    import matplotlib.pyplot as plt
    import starsmashertools.mpl.figure
    import starsmashertools.helpers.string
    # Importing axes.py registers our Axes classes for projections
    import starsmashertools.mpl.axes
    
    if FigureClass is None: FigureClass = matplotlib.figure.Figure
    elif isinstance(FigureClass, str):
        # Check for the string identifier in our custom Figure classes
        if FigureClass not in starsmashertools.mpl.figure.class_names.keys():
            s = starsmashertools.helpers.string.list_to_string(
                list(starsmashertools.mpl.figure.class_names.keys()),
                join = 'or',
            )
            raise ValueError("Unrecognized Figure class name '%s'. The available names are %s" % (FigureClass, s))
        FigureClass = starsmashertools.mpl.figure.class_names[FigureClass]

    # Search for custom Axes
    if isinstance(AxesClass, str):
        if AxesClass not in starsmashertools.mpl.axes.class_names.keys():
            s = starsmashertools.helpers.string.list_to_string(
                list(starsmashertools.mpl.axes.class_names.keys()),
                join = 'or',
            )
            raise ValueError("Unrecognized Axes class name '%s'. The available names are %s" % (AxesClass, s))
        
        if subplot_kw is None: subplot_kw = dict(projection=AxesClass)
        else: subplot_kw['projection'] = AxesClass

    if style is not None:
        starsmashertools.mpl.figure.update_style_sheet_directories()
        plt.style.use(style)
    
    fig, ax = plt.subplots(
        *args,
        FigureClass = FigureClass,
        subplot_kw = subplot_kw,
        **kwargs
    )
    fig.draw(renderer = fig.canvas.get_renderer())
    return fig, ax
